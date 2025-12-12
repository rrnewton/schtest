#!/usr/bin/env python3
"""
IRQ migration test runner with optional wprof tracing.

This script runs the IRQ migration test with optional wprof tracing support.
When --wprof is enabled, it:
1. Reserves a CPU (default: 175) from IRQ storm for the tracer
2. Launches wprof pinned to that CPU before the test
3. Runs the migration test
4. Collects the trace output

Topology notes (hard-coded for large AMD EPYC chip):
- CPU 0 (Die L#0, L3 L#0): Starting place for spinner task
- CPU 1: Control/destination for migration (intra-CCX "easy mode")
- CPU 175 (Die L#5): wprof core, different die/L3 from test cores

TODO: Add topology detection code to dynamically select appropriate CPUs.
      See util/system.rs for existing CPU topology code that could be ported.

Usage:
    sudo python3 scripts/run_irq_migration_test.py
    sudo python3 scripts/run_irq_migration_test.py --wprof
    sudo python3 scripts/run_irq_migration_test.py --wprof --wprof-cpu 175 --trace-output trace.pb
    sudo python3 scripts/run_irq_migration_test.py --trials 5 --duration 10
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# Hard-coded topology for AMD EPYC
# TODO: Detect dynamically from hwloc or /sys/devices/system/cpu/
DEFAULT_WPROF_CPU = 175  # Die L#5, different L3 from test cores


@dataclass
class MigrationResult:
    """Result from a single migration test run."""
    trial_num: int
    initial_cpu: int
    final_cpu: int
    migrated: bool
    migrated_to_control: bool
    scheduler: str
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "trial_num": self.trial_num,
            "initial_cpu": self.initial_cpu,
            "final_cpu": self.final_cpu,
            "migrated": self.migrated,
            "migrated_to_control": self.migrated_to_control,
            "scheduler": self.scheduler,
        }


def get_sched_ext_state() -> tuple[str, str]:
    """Return (state, scheduler_name) for sched_ext."""
    try:
        state = Path("/sys/kernel/sched_ext/state").read_text().strip()
        if state == "enabled":
            ops = Path("/sys/kernel/sched_ext/root/ops").read_text().strip()
            return state, ops
        return state, ""
    except FileNotFoundError:
        return "not_supported", ""


def get_current_scheduler() -> str:
    """Get the name of the current scheduler."""
    state, ops = get_sched_ext_state()
    if state == "enabled":
        return ops
    return "CFS"


def run_cmd(
    cmd: list[str],
    timeout: int = 600,
    check: bool = True,
    cwd: Optional[Path] = None,
    verbose: bool = True,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Run a command with timeout."""
    if verbose:
        print(f"  $ {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=check, cwd=cwd, env=run_env
    )


def build_schtest(repo_root: Path) -> Path:
    """Build schtest in release mode."""
    print("\nBuilding schtest...")
    run_cmd(["cargo", "build", "--release"], cwd=repo_root)

    schtest_path = repo_root / "target" / "release" / "schtest"
    if not schtest_path.exists():
        raise RuntimeError(f"schtest binary not found: {schtest_path}")

    return schtest_path


def start_wprof(cpu: int, duration_ms: int, trace_output: Path) -> subprocess.Popen:
    """Start wprof pinned to a specific CPU.

    Args:
        cpu: CPU ID to pin wprof to
        duration_ms: Trace duration in milliseconds
        trace_output: Path to write trace output

    Returns:
        The wprof subprocess (still running)
    """
    cmd = [
        "sudo", "taskset", "-c", str(cpu),
        "wprof", f"-d{duration_ms}", "-T", str(trace_output)
    ]
    print(f"  Starting wprof: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Give wprof time to initialize and start tracing
    time.sleep(0.5)

    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"wprof exited prematurely: {stderr}")

    print(f"  wprof running (PID {proc.pid}), tracing to {trace_output}")
    return proc


def wait_for_wprof(proc: subprocess.Popen, timeout: int = 30) -> tuple[str, str]:
    """Wait for wprof to complete and return its output."""
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return stdout, stderr


def run_migration_test(
    schtest_path: Path,
    duration: int = 10,
    reserved_cpu: Optional[int] = None,
) -> str:
    """Run the IRQ migration test and return output."""
    env = {
        "SCHTEST_IRQ_DURATION": str(duration),
    }
    if reserved_cpu is not None:
        env["SCHTEST_IRQ_RESERVE_TRACING_CORE"] = str(reserved_cpu)

    cmd = ["sudo", "-E", str(schtest_path), "--filter", "irq_migration"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=duration + 120,
        env={**os.environ, **env},
    )
    return result.stdout + result.stderr


def parse_migration_output(output: str) -> dict:
    """Parse migration test output to extract key metrics."""
    result = {}

    # Extract initial and final CPU
    import re

    initial_match = re.search(r"Initial CPU:\s+(\d+)", output)
    if initial_match:
        result["initial_cpu"] = int(initial_match.group(1))

    final_match = re.search(r"Final CPU:\s+(\d+)", output)
    if final_match:
        result["final_cpu"] = int(final_match.group(1))

    # Check for success/failure patterns
    if "SUCCESS" in output:
        result["migrated_to_control"] = True
        result["migrated"] = True
    elif "PARTIAL" in output:
        result["migrated_to_control"] = False
        result["migrated"] = True
    elif "FAILED" in output:
        result["migrated_to_control"] = False
        result["migrated"] = False

    return result


def run_single_trial(
    schtest_path: Path,
    trial_num: int,
    duration: int,
    wprof_enabled: bool = False,
    wprof_cpu: int = DEFAULT_WPROF_CPU,
    trace_output_dir: Optional[Path] = None,
    save_raw: bool = False,
    output_dir: Optional[Path] = None,
) -> MigrationResult:
    """Run a single migration test trial."""
    scheduler = get_current_scheduler()
    print(f"\n  Trial {trial_num}: scheduler={scheduler}")

    wprof_proc = None
    trace_file = None

    if wprof_enabled:
        # Calculate wprof duration: test duration + buffer (in ms)
        wprof_duration_ms = (duration + 5) * 1000

        if trace_output_dir:
            trace_file = trace_output_dir / f"trace_trial{trial_num:02d}.pb"
        else:
            trace_file = Path(f"trace_trial{trial_num:02d}.pb")

        try:
            wprof_proc = start_wprof(wprof_cpu, wprof_duration_ms, trace_file)
        except Exception as e:
            print(f"  WARNING: Failed to start wprof: {e}")
            wprof_proc = None

        # Give wprof a moment to start tracing
        time.sleep(0.5)

    # Run the test (with reserved CPU if wprof is enabled)
    reserved_cpu = wprof_cpu if wprof_enabled else None
    output = run_migration_test(schtest_path, duration, reserved_cpu)
    metrics = parse_migration_output(output)

    # Wait for wprof to finish if it was started
    if wprof_proc:
        print("  Waiting for wprof to finish...")
        stdout, stderr = wait_for_wprof(wprof_proc, timeout=duration + 30)
        if stderr:
            print(f"  wprof stderr: {stderr[:200]}")
        if trace_file and trace_file.exists():
            print(f"  Trace written to: {trace_file} ({trace_file.stat().st_size} bytes)")

    initial_cpu = metrics.get("initial_cpu", -1)
    final_cpu = metrics.get("final_cpu", -1)
    migrated = metrics.get("migrated", False)
    migrated_to_control = metrics.get("migrated_to_control", False)

    result = MigrationResult(
        trial_num=trial_num,
        initial_cpu=initial_cpu,
        final_cpu=final_cpu,
        migrated=migrated,
        migrated_to_control=migrated_to_control,
        scheduler=scheduler,
        raw_output=output if save_raw else "",
    )

    status = "SUCCESS" if migrated_to_control else ("PARTIAL" if migrated else "FAILED")
    print(f"    {status}: {initial_cpu} -> {final_cpu}")

    # Save raw output if requested
    if save_raw and output_dir:
        output_file = output_dir / f"migration_trial{trial_num:02d}.txt"
        output_file.write_text(output)

    return result


def print_summary(results: list[MigrationResult]):
    """Print summary of all trials."""
    print("\n")
    print("=" * 70)
    print("SUMMARY: IRQ Migration Test Results")
    print("=" * 70)
    print()

    success_count = sum(1 for r in results if r.migrated_to_control)
    partial_count = sum(1 for r in results if r.migrated and not r.migrated_to_control)
    failed_count = sum(1 for r in results if not r.migrated)

    print(f"Total trials: {len(results)}")
    print(f"  SUCCESS (migrated to control): {success_count}")
    print(f"  PARTIAL (migrated elsewhere):  {partial_count}")
    print(f"  FAILED (no migration):         {failed_count}")
    print()

    if results:
        print(f"{'Trial':>6} {'Initial':>8} {'Final':>8} {'Status':>12} {'Scheduler':>15}")
        print("-" * 55)
        for r in results:
            status = "SUCCESS" if r.migrated_to_control else ("PARTIAL" if r.migrated else "FAILED")
            print(f"{r.trial_num:>6} {r.initial_cpu:>8} {r.final_cpu:>8} {status:>12} {r.scheduler:>15}")


def main():
    parser = argparse.ArgumentParser(
        description="Run IRQ migration test with optional wprof tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trials", "-n", type=int, default=1,
        help="Number of trials to run (default: 1)"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=10,
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--wprof", action="store_true",
        help="Enable wprof tracing during the test"
    )
    parser.add_argument(
        "--wprof-cpu", type=int, default=DEFAULT_WPROF_CPU,
        help=f"CPU to pin wprof to (default: {DEFAULT_WPROF_CPU}, on different die/L3)"
    )
    parser.add_argument(
        "--trace-output", "-T", type=Path,
        help="Directory to save trace files (default: current directory)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        help="Directory to save raw test output files"
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Save raw test output to files"
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip building schtest (use existing binary)"
    )

    args = parser.parse_args()

    # Determine repo root
    repo_root = Path(__file__).parent.parent.resolve()
    print(f"Repository root: {repo_root}")

    # Set up output directories
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif args.save_raw:
        args.output_dir = repo_root / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving raw output to: {args.output_dir}")

    trace_output_dir = None
    if args.wprof:
        if args.trace_output:
            trace_output_dir = args.trace_output
            trace_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            trace_output_dir = Path(".")

    # Build schtest
    if args.skip_build:
        schtest_path = repo_root / "target" / "release" / "schtest"
        if not schtest_path.exists():
            print(f"ERROR: schtest not found at {schtest_path}")
            print("Run without --skip-build to build it first.")
            sys.exit(1)
    else:
        schtest_path = build_schtest(repo_root)

    # Check sudo access
    print("\nChecking sudo access...")
    try:
        subprocess.run(["sudo", "-n", "true"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("ERROR: Trials require sudo access. Please run with sudo or configure passwordless sudo.")
        sys.exit(1)

    # Report current scheduler state
    state, ops = get_sched_ext_state()
    print(f"\nScheduler state: {state}")
    if ops:
        print(f"Current scheduler: {ops}")
    else:
        print("Current scheduler: CFS (default)")

    # Run trials
    print(f"\n{'='*70}")
    print(f"Running {args.trials} trial(s), {args.duration}s each")
    if args.wprof:
        print(f"wprof enabled, pinned to CPU {args.wprof_cpu}")
    print(f"{'='*70}")

    results = []
    for trial in range(1, args.trials + 1):
        result = run_single_trial(
            schtest_path,
            trial,
            args.duration,
            wprof_enabled=args.wprof,
            wprof_cpu=args.wprof_cpu,
            trace_output_dir=trace_output_dir,
            save_raw=args.save_raw,
            output_dir=args.output_dir,
        )
        results.append(result)

        # Brief pause between trials
        if trial < args.trials:
            time.sleep(1)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
