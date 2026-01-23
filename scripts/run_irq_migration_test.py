#!/usr/bin/env python3
"""
IRQ migration test runner with scheduler orchestration and monitoring.

This script runs the IRQ migration test with optional:
- Scheduler management (start/stop scx_lavd or other schedulers)
- LAVD monitoring to track LAT_CAP values per CPU
- wprof tracing support

Usage:
    # Basic test with current scheduler
    sudo python3 scripts/run_irq_migration_test.py

    # Start LAVD scheduler and run test
    sudo python3 scripts/run_irq_migration_test.py --start-scheduler=lavd

    # Start LAVD with monitoring to track LAT_CAP
    sudo python3 scripts/run_irq_migration_test.py --start-scheduler=lavd --lavd-monitoring

    # With wprof tracing
    sudo python3 scripts/run_irq_migration_test.py --start-scheduler=lavd --wprof
"""

import argparse
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Hard-coded topology for AMD EPYC
DEFAULT_WPROF_CPU = 175


@dataclass
class LatCapStats:
    """Statistics for LAT_CAP values on a single CPU."""
    cpu_id: int
    samples: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def first(self) -> Optional[int]:
        return self.samples[0] if self.samples else None

    @property
    def last(self) -> Optional[int]:
        return self.samples[-1] if self.samples else None

    @property
    def min(self) -> Optional[int]:
        return min(self.samples) if self.samples else None

    @property
    def max(self) -> Optional[int]:
        return max(self.samples) if self.samples else None

    @property
    def avg(self) -> Optional[float]:
        return sum(self.samples) / len(self.samples) if self.samples else None


@dataclass
class MigrationResult:
    """Result from a single migration test run."""
    trial_num: int
    initial_cpu: int
    final_cpu: int
    final_cpu_b: int  # For ping-pong probes
    migrated: bool
    migrated_to_control: bool
    scheduler: str
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "trial_num": self.trial_num,
            "initial_cpu": self.initial_cpu,
            "final_cpu": self.final_cpu,
            "final_cpu_b": self.final_cpu_b,
            "migrated": self.migrated,
            "migrated_to_control": self.migrated_to_control,
            "scheduler": self.scheduler,
        }


class LavdMonitor:
    """Monitor LAVD scheduler output and track LAT_CAP per CPU."""

    def __init__(self, scx_lavd_path: Path, nr_samples: int = 32):
        self.scx_lavd_path = scx_lavd_path
        self.nr_samples = nr_samples
        self.process: Optional[subprocess.Popen] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.lat_cap_stats: dict[int, LatCapStats] = defaultdict(lambda: LatCapStats(cpu_id=-1))
        self.lock = threading.Lock()
        self.running = False
        self.output_lines: list[str] = []

    def _parse_monitor_line(self, line: str):
        """Parse a single monitor output line and extract CPU and LAT_CAP."""
        # Monitor format: | MSEQ | PID | COMM | STAT | CPU | ... | LAT_CAP | ...
        # Fields are separated by |
        if not line.startswith("|"):
            return

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 24:  # Need at least up to LAT_CAP
            return

        # Skip header lines
        if parts[1] == "MSEQ" or parts[5] == "CPU":
            return

        try:
            # CPU is at index 5, LAT_CAP is at index 24
            cpu_str = parts[5]
            lat_cap_str = parts[24]

            # Skip if not numeric
            if not cpu_str.isdigit():
                return

            cpu_id = int(cpu_str)
            lat_cap = int(lat_cap_str)

            with self.lock:
                if self.lat_cap_stats[cpu_id].cpu_id == -1:
                    self.lat_cap_stats[cpu_id] = LatCapStats(cpu_id=cpu_id)
                self.lat_cap_stats[cpu_id].samples.append(lat_cap)

        except (ValueError, IndexError):
            pass  # Skip malformed lines

    def _monitor_loop(self):
        """Background thread to read monitor output."""
        try:
            for line in iter(self.process.stdout.readline, ""):
                if not self.running:
                    break
                line = line.rstrip()
                self.output_lines.append(line)
                self._parse_monitor_line(line)
        except Exception as e:
            print(f"Monitor thread error: {e}")

    def start(self):
        """Start the LAVD monitor process."""
        cmd = [
            "sudo", str(self.scx_lavd_path),
            f"--monitor-sched-samples={self.nr_samples}"
        ]
        print(f"  Starting LAVD monitor: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        # Give it a moment to start
        time.sleep(0.5)

        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"LAVD monitor exited prematurely: {stderr}")

        print(f"  LAVD monitor running (PID {self.process.pid})")

    def stop(self) -> dict[int, LatCapStats]:
        """Stop the monitor and return collected stats."""
        self.running = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

        with self.lock:
            return dict(self.lat_cap_stats)


class SchedulerManager:
    """Manage starting and stopping schedulers."""

    # Relative paths to try for scheduler binaries (relative to repo root's parent)
    SCHEDULER_SEARCH_PATHS = [
        "scx/target/release",      # ../scx/target/release (sibling directory)
        "../scx/target/release",   # Alternative sibling path
    ]

    SCHEDULER_BINARIES = {
        "lavd": "scx_lavd",
        "rusty": "scx_rusty",
        "bpfland": "scx_bpfland",
    }

    SCHEDULER_ARGS = {
        "lavd": ["--performance"],
        "rusty": [],
        "bpfland": [],
    }

    def __init__(self, repo_root: Path, scheduler_name: str):
        self.repo_root = repo_root
        self.scheduler_name = scheduler_name
        self.process: Optional[subprocess.Popen] = None

        if scheduler_name not in self.SCHEDULER_BINARIES:
            raise ValueError(f"Unknown scheduler: {scheduler_name}. "
                           f"Known: {list(self.SCHEDULER_BINARIES.keys())}")

        binary_name = self.SCHEDULER_BINARIES[scheduler_name]
        self.scheduler_path = self._find_scheduler_binary(binary_name)

    def _find_scheduler_binary(self, binary_name: str) -> Path:
        """Find the scheduler binary in known locations."""
        # Try paths relative to repo root's parent (for sibling scx directory)
        parent_dir = self.repo_root.parent
        for search_path in self.SCHEDULER_SEARCH_PATHS:
            candidate = parent_dir / search_path / binary_name
            if candidate.exists():
                return candidate

        # Try paths relative to repo root itself (in case scx is inside)
        for search_path in self.SCHEDULER_SEARCH_PATHS:
            candidate = self.repo_root / search_path / binary_name
            if candidate.exists():
                return candidate

        # Build a helpful error message
        searched = []
        for search_path in self.SCHEDULER_SEARCH_PATHS:
            searched.append(str(parent_dir / search_path / binary_name))
            searched.append(str(self.repo_root / search_path / binary_name))

        raise FileNotFoundError(
            f"Scheduler binary '{binary_name}' not found. Searched:\n" +
            "\n".join(f"  - {p}" for p in searched)
        )

    def start(self):
        """Start the scheduler."""
        args = self.SCHEDULER_ARGS.get(self.scheduler_name, [])
        cmd = ["sudo", str(self.scheduler_path)] + args

        print(f"  Starting scheduler: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for scheduler to initialize
        time.sleep(1.0)

        if self.process.poll() is not None:
            stdout = self.process.stdout.read()
            stderr = self.process.stderr.read()
            raise RuntimeError(f"Scheduler exited prematurely:\nstdout: {stdout}\nstderr: {stderr}")

        # Verify sched_ext is enabled
        state, ops = get_sched_ext_state()
        if state != "enabled":
            raise RuntimeError(f"Scheduler failed to enable sched_ext (state: {state})")

        print(f"  Scheduler {ops} running (PID {self.process.pid})")

    def is_alive(self) -> bool:
        """Check if the scheduler process is still running."""
        if not self.process:
            return False
        return self.process.poll() is None

    def check_alive(self):
        """Check if scheduler is alive and raise an error if it died."""
        if not self.process:
            return  # No scheduler was started by us

        exit_code = self.process.poll()
        if exit_code is not None:
            # Scheduler died - collect output for diagnostics
            stdout = self.process.stdout.read() if self.process.stdout else ""
            stderr = self.process.stderr.read() if self.process.stderr else ""
            self.process = None  # Mark as dead

            error_msg = f"Scheduler died unexpectedly (exit code {exit_code})"
            if stdout:
                error_msg += f"\nstdout: {stdout[-2000:]}"  # Last 2000 chars
            if stderr:
                error_msg += f"\nstderr: {stderr[-2000:]}"
            raise RuntimeError(error_msg)

        # Also verify sched_ext is still enabled
        state, ops = get_sched_ext_state()
        if state != "enabled":
            raise RuntimeError(f"Scheduler process alive but sched_ext not enabled (state: {state})")

    def stop(self):
        """Stop the scheduler gracefully."""
        if not self.process:
            return

        print(f"  Stopping scheduler (PID {self.process.pid})...")

        # Send SIGTERM first
        self.process.terminate()

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("  Scheduler didn't stop gracefully, sending SIGKILL...")
            self.process.kill()
            self.process.wait()

        self.process = None

        # Verify sched_ext is disabled
        time.sleep(0.5)
        state, _ = get_sched_ext_state()
        if state == "enabled":
            print("  WARNING: sched_ext still enabled after stopping scheduler")


def get_kernel_version() -> tuple[int, int, int]:
    """Get the kernel version as a tuple (major, minor, patch)."""
    release = platform.release()
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", release)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return (0, 0, 0)


def get_default_scheduler_name() -> str:
    """Get the name of the default Linux scheduler for the running kernel."""
    major, minor, _ = get_kernel_version()
    if major > 6 or (major == 6 and minor >= 6):
        return "EEVDF"
    return "CFS"


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
    return get_default_scheduler_name()


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
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd, env=run_env
    )
    if result.returncode != 0 and check:
        print(f"  Command failed with exit code {result.returncode}")
        if result.stdout:
            print(f"  stdout: {result.stdout[:2000]}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:2000]}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def build_schtest(repo_root: Path) -> Path:
    """Build schtest in release mode."""
    print("\nBuilding schtest...")
    run_cmd(["cargo", "build", "--release"], cwd=repo_root)

    schtest_path = repo_root / "target" / "release" / "schtest"
    if not schtest_path.exists():
        raise RuntimeError(f"schtest binary not found: {schtest_path}")

    return schtest_path


def start_wprof(cpu: int, duration_ms: int, trace_output: Path) -> subprocess.Popen:
    """Start wprof pinned to a specific CPU."""
    cmd = [
        "sudo", "taskset", "-c", str(cpu),
        "wprof", f"-d{duration_ms}", "-T", str(trace_output)
    ]
    print(f"  Starting wprof: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

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

    # Extract initial CPU
    initial_match = re.search(r"Initial CPU:\s+(\d+)", output)
    if initial_match:
        result["initial_cpu"] = int(initial_match.group(1))

    # Extract final CPU (probe A or single worker)
    final_match = re.search(r"Final CPU(?:\s+\(probe A\))?:\s+(\d+)", output)
    if final_match:
        result["final_cpu"] = int(final_match.group(1))

    # Extract final CPU for probe B (ping-pong mode)
    final_b_match = re.search(r"Final CPU \(probe B\):\s+(\d+)", output)
    if final_b_match:
        result["final_cpu_b"] = int(final_b_match.group(1))
    else:
        result["final_cpu_b"] = result.get("final_cpu", -1)

    # Extract control CPUs
    control_match = re.search(r"Control CPUs:\s+\[([^\]]+)\]", output)
    if control_match:
        result["control_cpus"] = [int(x.strip()) for x in control_match.group(1).split(",")]

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


def parse_victim_cpus_from_output(output: str) -> list[int]:
    """Extract victim CPU IDs from test output."""
    # Look for lines like "Timer interrupts enabled on CPU 0 at..."
    victim_cpus = []
    for match in re.finditer(r"Timer interrupts enabled on CPU (\d+)", output):
        victim_cpus.append(int(match.group(1)))
    return victim_cpus


def run_single_trial(
    schtest_path: Path,
    trial_num: int,
    total_trials: int,
    duration: int,
    wprof_enabled: bool = False,
    wprof_cpu: int = DEFAULT_WPROF_CPU,
    trace_output_dir: Optional[Path] = None,
    save_raw: bool = False,
    output_dir: Optional[Path] = None,
    lavd_monitor: Optional[LavdMonitor] = None,
) -> tuple[MigrationResult, str]:
    """Run a single migration test trial. Returns (result, raw_output)."""
    scheduler = get_current_scheduler()
    print(f"\n  Trial {trial_num} of {total_trials}: scheduler={scheduler}")

    wprof_proc = None
    trace_file = None

    if wprof_enabled:
        wprof_duration_ms = (duration + 5) * 1000
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if trace_output_dir:
            trace_file = trace_output_dir / f"trace_{timestamp}_trial{trial_num:02d}.pb"
        else:
            trace_file = Path(f"trace_{timestamp}_trial{trial_num:02d}.pb")

        try:
            wprof_proc = start_wprof(wprof_cpu, wprof_duration_ms, trace_file)
        except Exception as e:
            print(f"  WARNING: Failed to start wprof: {e}")
            wprof_proc = None

        time.sleep(0.5)

    # Run the test
    reserved_cpu = wprof_cpu if wprof_enabled else None
    output = run_migration_test(schtest_path, duration, reserved_cpu)
    metrics = parse_migration_output(output)

    # Wait for wprof if started
    if wprof_proc:
        print("  Waiting for wprof to finish...")
        stdout, stderr = wait_for_wprof(wprof_proc, timeout=duration + 30)
        if stderr:
            print(f"  wprof stderr: {stderr[:200]}")
        if trace_file and trace_file.exists():
            print(f"  Trace written to: {trace_file} ({trace_file.stat().st_size} bytes)")

    initial_cpu = metrics.get("initial_cpu", -1)
    final_cpu = metrics.get("final_cpu", -1)
    final_cpu_b = metrics.get("final_cpu_b", -1)
    migrated = metrics.get("migrated", False)
    migrated_to_control = metrics.get("migrated_to_control", False)

    result = MigrationResult(
        trial_num=trial_num,
        initial_cpu=initial_cpu,
        final_cpu=final_cpu,
        final_cpu_b=final_cpu_b,
        migrated=migrated,
        migrated_to_control=migrated_to_control,
        scheduler=scheduler,
        raw_output=output if save_raw else "",
    )

    status = "SUCCESS" if migrated_to_control else ("PARTIAL" if migrated else "FAILED")
    print(f"    {status}: {initial_cpu} -> A:{final_cpu}, B:{final_cpu_b}")

    # Save raw output if requested
    if save_raw and output_dir:
        output_file = output_dir / f"migration_trial{trial_num:02d}.txt"
        output_file.write_text(output)

    return result, output


def print_lat_cap_summary(
    lat_cap_stats: dict[int, LatCapStats],
    victim_cpus: list[int],
    control_cpus: list[int],
):
    """Print LAT_CAP statistics summary."""
    print("\n" + "=" * 80)
    print("LAT_CAP Statistics by CPU")
    print("=" * 80)

    if not lat_cap_stats:
        print("  No LAT_CAP data collected")
        return

    # Print header
    print(f"{'CPU':>6} {'Role':>10} {'Samples':>8} {'First':>8} {'Last':>8} {'Min':>8} {'Max':>8} {'Avg':>10}")
    print("-" * 80)

    # Sort by CPU ID
    for cpu_id in sorted(lat_cap_stats.keys()):
        stats = lat_cap_stats[cpu_id]
        if stats.count == 0:
            continue

        if cpu_id in control_cpus:
            role = "CONTROL"
        elif cpu_id in victim_cpus:
            role = "VICTIM"
        else:
            role = "OTHER"

        print(f"{cpu_id:>6} {role:>10} {stats.count:>8} {stats.first or 0:>8} {stats.last or 0:>8} "
              f"{stats.min or 0:>8} {stats.max or 0:>8} {stats.avg or 0:>10.1f}")


def validate_lat_cap_results(
    lat_cap_stats: dict[int, LatCapStats],
    victim_cpus: list[int],
    control_cpus: list[int],
) -> bool:
    """
    Validate that LAT_CAP behaves as expected:
    1. Victim CPUs should have decreased LAT_CAP from start
    2. Control CPU should maintain higher min/avg LAT_CAP than victims
    """
    print("\n" + "=" * 80)
    print("LAT_CAP Validation")
    print("=" * 80)

    if not lat_cap_stats:
        print("  No LAT_CAP data to validate")
        return False

    all_passed = True

    # Compute aggregate stats for victim and control CPUs
    victim_stats = [lat_cap_stats.get(cpu) for cpu in victim_cpus if cpu in lat_cap_stats]
    control_stats = [lat_cap_stats.get(cpu) for cpu in control_cpus if cpu in lat_cap_stats]

    victim_stats = [s for s in victim_stats if s and s.count > 0]
    control_stats = [s for s in control_stats if s and s.count > 0]

    if not victim_stats:
        print("  WARNING: No LAT_CAP data for victim CPUs")
        return False

    if not control_stats:
        print("  WARNING: No LAT_CAP data for control CPUs")
        return False

    # Check 1: Victim CPUs should show decreased LAT_CAP
    print("\n  Check 1: Victim CPUs LAT_CAP decreased from start")
    victims_decreased = 0
    for stats in victim_stats:
        if stats.first and stats.last:
            decreased = stats.last < stats.first
            if decreased:
                victims_decreased += 1
            print(f"    CPU {stats.cpu_id}: {stats.first} -> {stats.last} "
                  f"({'✓ decreased' if decreased else '✗ not decreased'})")

    if victims_decreased > 0:
        print(f"  ✓ {victims_decreased}/{len(victim_stats)} victim CPUs showed LAT_CAP decrease")
    else:
        print(f"  ✗ No victim CPUs showed LAT_CAP decrease")
        all_passed = False

    # Check 2: Control CPUs should have higher min LAT_CAP than victim average
    print("\n  Check 2: Control CPUs maintain higher LAT_CAP than victims")

    victim_avg_min = sum(s.min or 0 for s in victim_stats) / len(victim_stats) if victim_stats else 0
    victim_avg_avg = sum(s.avg or 0 for s in victim_stats) / len(victim_stats) if victim_stats else 0

    control_min_of_mins = min(s.min or 0 for s in control_stats) if control_stats else 0
    control_avg_of_avgs = sum(s.avg or 0 for s in control_stats) / len(control_stats) if control_stats else 0

    print(f"    Victim CPUs:  avg(min)={victim_avg_min:.1f}, avg(avg)={victim_avg_avg:.1f}")
    print(f"    Control CPUs: min(min)={control_min_of_mins:.1f}, avg(avg)={control_avg_of_avgs:.1f}")

    if control_min_of_mins > victim_avg_min:
        print(f"  ✓ Control min LAT_CAP ({control_min_of_mins:.0f}) > victim avg min ({victim_avg_min:.0f})")
    else:
        print(f"  ✗ Control min LAT_CAP ({control_min_of_mins:.0f}) <= victim avg min ({victim_avg_min:.0f})")
        all_passed = False

    if control_avg_of_avgs > victim_avg_avg:
        print(f"  ✓ Control avg LAT_CAP ({control_avg_of_avgs:.0f}) > victim avg ({victim_avg_avg:.0f})")
    else:
        print(f"  ✗ Control avg LAT_CAP ({control_avg_of_avgs:.0f}) <= victim avg ({victim_avg_avg:.0f})")
        all_passed = False

    print("\n" + "-" * 80)
    if all_passed:
        print("  ✓ All LAT_CAP validations PASSED")
    else:
        print("  ✗ Some LAT_CAP validations FAILED")

    return all_passed


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
        print(f"{'Trial':>6} {'Initial':>8} {'Final_A':>8} {'Final_B':>8} {'Status':>12} {'Scheduler':>15}")
        print("-" * 65)
        for r in results:
            status = "SUCCESS" if r.migrated_to_control else ("PARTIAL" if r.migrated else "FAILED")
            print(f"{r.trial_num:>6} {r.initial_cpu:>8} {r.final_cpu:>8} {r.final_cpu_b:>8} {status:>12} {r.scheduler:>15}")


def main():
    parser = argparse.ArgumentParser(
        description="Run IRQ migration test with scheduler orchestration and monitoring",
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
        "--start-scheduler", type=str, default=None,
        metavar="NAME",
        help="Start a scheduler before the test (e.g., 'lavd', 'rusty', 'bpfland')"
    )
    parser.add_argument(
        "--lavd-monitoring", action="store_true",
        help="Enable LAVD monitoring to track LAT_CAP values (requires --start-scheduler=lavd)"
    )
    parser.add_argument(
        "--monitor-samples", type=int, default=32,
        help="Number of scheduling samples per second for LAVD monitor (default: 32)"
    )
    parser.add_argument(
        "--wprof", action="store_true",
        help="Enable wprof tracing during the test"
    )
    parser.add_argument(
        "--wprof-cpu", type=int, default=DEFAULT_WPROF_CPU,
        help=f"CPU to pin wprof to (default: {DEFAULT_WPROF_CPU})"
    )
    parser.add_argument(
        "--trace-output", "-T", type=Path,
        help="Directory to save trace files"
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

    # Validate args
    if args.lavd_monitoring and args.start_scheduler != "lavd":
        print("ERROR: --lavd-monitoring requires --start-scheduler=lavd")
        sys.exit(1)

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
        print("ERROR: Test requires sudo access. Please run with sudo or configure passwordless sudo.")
        sys.exit(1)

    # Start scheduler if requested
    scheduler_mgr = None
    if args.start_scheduler:
        print(f"\nStarting scheduler: {args.start_scheduler}")
        try:
            scheduler_mgr = SchedulerManager(repo_root, args.start_scheduler)
            scheduler_mgr.start()
        except Exception as e:
            print(f"ERROR: Failed to start scheduler: {e}")
            sys.exit(1)

    # Start LAVD monitor if requested
    lavd_monitor = None
    if args.lavd_monitoring:
        print("\nStarting LAVD monitor...")
        try:
            # Use the same path as the scheduler manager found
            if scheduler_mgr:
                scx_lavd_path = scheduler_mgr.scheduler_path
            else:
                # Fall back to searching for it
                scx_lavd_path = SchedulerManager(repo_root, "lavd").scheduler_path
            lavd_monitor = LavdMonitor(scx_lavd_path, args.monitor_samples)
            lavd_monitor.start()
        except Exception as e:
            print(f"ERROR: Failed to start LAVD monitor: {e}")
            if scheduler_mgr:
                scheduler_mgr.stop()
            sys.exit(1)

    # Report current scheduler state
    state, ops = get_sched_ext_state()
    print(f"\nScheduler state: {state}")
    if ops:
        print(f"Current scheduler: {ops}")
    else:
        print(f"Current scheduler: {get_default_scheduler_name()} (default)")

    # Run trials
    print(f"\n{'='*70}")
    print(f"Running {args.trials} trial(s), {args.duration}s each")
    if args.lavd_monitoring:
        print(f"LAVD monitoring enabled ({args.monitor_samples} samples/sec)")
    if args.wprof:
        print(f"wprof enabled, pinned to CPU {args.wprof_cpu}")
    print(f"{'='*70}")

    results = []
    all_output = ""
    victim_cpus = []
    control_cpus = []

    try:
        for trial in range(1, args.trials + 1):
            result, output = run_single_trial(
                schtest_path,
                trial,
                args.trials,
                args.duration,
                wprof_enabled=args.wprof,
                wprof_cpu=args.wprof_cpu,
                trace_output_dir=trace_output_dir,
                save_raw=args.save_raw,
                output_dir=args.output_dir,
                lavd_monitor=lavd_monitor,
            )
            results.append(result)
            all_output += output

            # Check scheduler is still alive after each trial
            if scheduler_mgr:
                scheduler_mgr.check_alive()

            # Extract victim and control CPUs from first trial
            if trial == 1:
                victim_cpus = parse_victim_cpus_from_output(output)
                metrics = parse_migration_output(output)
                control_cpus = metrics.get("control_cpus", [])

            # Brief pause between trials
            if trial < args.trials:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except RuntimeError as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)
    finally:
        # Stop LAVD monitor and collect stats
        lat_cap_stats = {}
        if lavd_monitor:
            print("\nStopping LAVD monitor...")
            lat_cap_stats = lavd_monitor.stop()

        # Stop scheduler
        if scheduler_mgr:
            print("\nStopping scheduler...")
            scheduler_mgr.stop()

    # Print results
    print_summary(results)

    # Print and validate LAT_CAP stats if monitoring was enabled
    if lat_cap_stats:
        print_lat_cap_summary(lat_cap_stats, victim_cpus, control_cpus)
        validate_lat_cap_results(lat_cap_stats, victim_cpus, control_cpus)


if __name__ == "__main__":
    main()
