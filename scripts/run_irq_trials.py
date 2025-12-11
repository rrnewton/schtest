#!/usr/bin/env python3
"""
IRQ accounting trial runner for comparing LAVD scheduler versions.

This script runs the IRQ disruption test across multiple LAVD versions to measure
the impact of IRQ time accounting on scheduler fairness. It builds different versions
of scx_lavd, runs N trials with each, and reports statistical distributions.

LAVD Versions tested:
- EEVDF (baseline): Native kernel scheduler without sched_ext
- main: ac58df714beda5f7d7c0a5ccfd9c493023600f00
- sans_irq_accounting: 35999abe5f39acde0bd7a38097c2fd302972ace8
- irq (new load balancing): 424b17be3e6bc613ec86913d7ee1024ee2222a20

Usage:
    sudo python3 scripts/run_irq_trials.py --trials 10
    sudo python3 scripts/run_irq_trials.py --trials 5 --skip-build  # Use existing binaries
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# SCX commits for different LAVD versions
LAVD_VERSIONS = {
    "main": "ac58df714beda5f7d7c0a5ccfd9c493023600f00",
    "sans_irq_accounting": "35999abe5f39acde0bd7a38097c2fd302972ace8",
    "irq": "424b17be3e6bc613ec86913d7ee1024ee2222a20",
}

# Directory to store built binaries (relative to repo root)
BIN_DIR = "bin"


@dataclass
class TrialResult:
    """Result from a single trial run."""
    scheduler: str
    trial_num: int
    victim_scheduled_secs: float
    control_scheduled_secs: float
    victim_utilization_p50: float
    control_utilization_p50: float
    gap_percent: float  # (control - victim) / control * 100
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "scheduler": self.scheduler,
            "trial_num": self.trial_num,
            "victim_scheduled_secs": self.victim_scheduled_secs,
            "control_scheduled_secs": self.control_scheduled_secs,
            "victim_utilization_p50": self.victim_utilization_p50,
            "control_utilization_p50": self.control_utilization_p50,
            "gap_percent": self.gap_percent,
        }


@dataclass
class SchedulerStats:
    """Aggregated statistics for a scheduler across all trials."""
    scheduler: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def gap_percentages(self) -> list[float]:
        return [t.gap_percent for t in self.trials]

    @property
    def mean_gap(self) -> float:
        gaps = self.gap_percentages
        return statistics.mean(gaps) if gaps else 0.0

    @property
    def stdev_gap(self) -> float:
        gaps = self.gap_percentages
        return statistics.stdev(gaps) if len(gaps) > 1 else 0.0

    @property
    def min_gap(self) -> float:
        gaps = self.gap_percentages
        return min(gaps) if gaps else 0.0

    @property
    def max_gap(self) -> float:
        gaps = self.gap_percentages
        return max(gaps) if gaps else 0.0

    @property
    def median_gap(self) -> float:
        gaps = self.gap_percentages
        return statistics.median(gaps) if gaps else 0.0

    def to_dict(self) -> dict:
        return {
            "scheduler": self.scheduler,
            "n_trials": len(self.trials),
            "gap_mean": self.mean_gap,
            "gap_stdev": self.stdev_gap,
            "gap_min": self.min_gap,
            "gap_max": self.max_gap,
            "gap_median": self.median_gap,
            "trials": [t.to_dict() for t in self.trials],
        }


def run_cmd(
    cmd: list[str],
    timeout: int = 600,
    check: bool = True,
    cwd: Optional[Path] = None,
    verbose: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command with timeout."""
    if verbose:
        print(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=check, cwd=cwd
    )


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


def stop_all_scx_schedulers():
    """Stop any running sched_ext scheduler."""
    state, ops = get_sched_ext_state()
    if state == "enabled":
        print(f"  Stopping running scheduler: {ops}")
        subprocess.run(["sudo", "pkill", "-9", "-f", "scx_"], check=False)
        time.sleep(2)
        state, _ = get_sched_ext_state()
        if state == "enabled":
            raise RuntimeError("Failed to stop sched_ext scheduler")


def clone_scx_repo(work_dir: Path) -> Path:
    """Clone the scx repository if not already present."""
    scx_dir = work_dir / "scx"
    if scx_dir.exists():
        print(f"  SCX repo already exists at {scx_dir}")
        # Fetch latest
        run_cmd(["git", "fetch", "--all"], cwd=scx_dir, verbose=False)
        return scx_dir

    print("  Cloning scx repository...")
    run_cmd(["git", "clone", "https://github.com/sched-ext/scx.git", str(scx_dir)])
    return scx_dir


def build_lavd_at_commit(scx_dir: Path, commit: str, output_name: str, bin_dir: Path) -> Path:
    """Build scx_lavd at a specific commit and copy to bin_dir."""
    output_path = bin_dir / output_name
    if output_path.exists():
        print(f"  Binary already exists: {output_path}")
        return output_path

    print(f"\n  Building LAVD at commit {commit[:12]}...")

    # Save original branch/commit to restore later
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=scx_dir, capture_output=True, text=True
    )
    original_ref = result.stdout.strip()
    if original_ref == "HEAD":
        # Detached HEAD, get the commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=scx_dir, capture_output=True, text=True
        )
        original_ref = result.stdout.strip()

    try:
        # Clean any previous build
        build_dir = scx_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)

        # Checkout the commit
        run_cmd(["git", "checkout", commit], cwd=scx_dir)
        run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=scx_dir, verbose=False)

        # Build using cargo (scx_lavd is in scheds/rust/scx_lavd)
        lavd_dir = scx_dir / "scheds" / "rust" / "scx_lavd"
        print("  Running cargo build --release...")
        run_cmd(
            ["cargo", "build", "--release"],
            cwd=lavd_dir,
            timeout=600
        )

        # Find the built binary
        lavd_path = scx_dir / "target" / "release" / "scx_lavd"
        if not lavd_path.exists():
            raise RuntimeError(f"scx_lavd binary not found after build at commit {commit}")

        # Copy to bin directory
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lavd_path, output_path)
        output_path.chmod(0o755)
        print(f"  Copied binary to: {output_path}")

        return output_path

    finally:
        # Restore original branch/commit
        print(f"  Restoring original ref: {original_ref}")
        subprocess.run(["git", "checkout", original_ref], cwd=scx_dir, capture_output=True)


def build_all_lavd_versions(repo_root: Path, work_dir: Path) -> dict[str, Path]:
    """Build all LAVD versions and return paths."""
    bin_dir = repo_root / BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    scx_dir = clone_scx_repo(work_dir)
    binaries = {}

    for name, commit in LAVD_VERSIONS.items():
        output_name = f"scx_lavd_{name}"
        try:
            binaries[name] = build_lavd_at_commit(scx_dir, commit, output_name, bin_dir)
        except Exception as e:
            print(f"  WARNING: Failed to build {name}: {e}")

    return binaries


def build_schtest(repo_root: Path) -> Path:
    """Build schtest in release mode."""
    print("\nBuilding schtest...")
    run_cmd(["cargo", "build", "--release"], cwd=repo_root)

    schtest_path = repo_root / "target" / "release" / "schtest"
    if not schtest_path.exists():
        raise RuntimeError(f"schtest binary not found: {schtest_path}")

    return schtest_path


def start_lavd(lavd_path: Path, extra_args: list[str] = None) -> subprocess.Popen:
    """Start scx_lavd scheduler."""
    cmd = ["sudo", str(lavd_path)] + (extra_args or [])
    print(f"  Starting LAVD: {lavd_path.name}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    time.sleep(3)  # Wait for scheduler to initialize

    state, ops = get_sched_ext_state()
    if state != "enabled" or "lavd" not in ops.lower():
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            f"Failed to start LAVD. State: {state}, Ops: {ops}\nstderr: {stderr}"
        )

    print(f"  LAVD running: {ops}")
    return proc


def stop_lavd(proc: subprocess.Popen):
    """Stop scx_lavd scheduler."""
    subprocess.run(["sudo", "pkill", "-f", "scx_lavd"], check=False)
    time.sleep(1)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_irq_test(schtest_path: Path, duration: int = 10) -> str:
    """Run the IRQ test and return output."""
    env = os.environ.copy()
    env["SCHTEST_IRQ_DURATION"] = str(duration)

    cmd = ["sudo", "-E", str(schtest_path), "--filter", "irq"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=duration + 60, env=env
    )
    return result.stdout + result.stderr


def parse_test_output(output: str) -> dict:
    """Parse test output to extract metrics."""
    result = {}

    # Extract scheduled time from worker summaries
    victim_match = re.search(
        r"Worker 'victim' Summary.*?Scheduled time:\s+([\d.]+)\s+seconds",
        output,
        re.DOTALL,
    )
    if victim_match:
        result["victim_scheduled_secs"] = float(victim_match.group(1))

    control_match = re.search(
        r"Worker 'control' Summary.*?Scheduled time:\s+([\d.]+)\s+seconds",
        output,
        re.DOTALL,
    )
    if control_match:
        result["control_scheduled_secs"] = float(control_match.group(1))

    # Extract utilization percentiles (p50)
    victim_util_match = re.search(
        r"Worker 'victim' Summary.*?Utilization percentiles:.*?p50:\s+([\d.]+)%",
        output,
        re.DOTALL,
    )
    if victim_util_match:
        result["victim_utilization_p50"] = float(victim_util_match.group(1))

    control_util_match = re.search(
        r"Worker 'control' Summary.*?Utilization percentiles:.*?p50:\s+([\d.]+)%",
        output,
        re.DOTALL,
    )
    if control_util_match:
        result["control_utilization_p50"] = float(control_util_match.group(1))

    return result


def run_single_trial(
    scheduler_name: str,
    trial_num: int,
    schtest_path: Path,
    lavd_path: Optional[Path],
    duration: int,
    save_raw: bool = False,
    output_dir: Optional[Path] = None,
) -> TrialResult:
    """Run a single trial and return the result."""
    print(f"\n  Trial {trial_num}: {scheduler_name}")

    # Run the test
    output = run_irq_test(schtest_path, duration)
    metrics = parse_test_output(output)

    victim_secs = metrics.get("victim_scheduled_secs", 0)
    control_secs = metrics.get("control_scheduled_secs", 0)

    # Calculate gap: (control - victim) / control * 100
    # Positive gap means control got more time (expected due to IRQ load on victim)
    gap_pct = 0.0
    if control_secs > 0:
        gap_pct = (control_secs - victim_secs) / control_secs * 100

    result = TrialResult(
        scheduler=scheduler_name,
        trial_num=trial_num,
        victim_scheduled_secs=victim_secs,
        control_scheduled_secs=control_secs,
        victim_utilization_p50=metrics.get("victim_utilization_p50", 0),
        control_utilization_p50=metrics.get("control_utilization_p50", 0),
        gap_percent=gap_pct,
        raw_output=output if save_raw else "",
    )

    print(
        f"    victim={victim_secs:.3f}s, control={control_secs:.3f}s, gap={gap_pct:.2f}%"
    )

    # Save raw output if requested
    if save_raw and output_dir:
        output_file = output_dir / f"{scheduler_name}_trial{trial_num:02d}.txt"
        output_file.write_text(output)

    return result


def run_all_trials(
    schtest_path: Path,
    lavd_binaries: dict[str, Path],
    n_trials: int,
    duration: int,
    save_raw: bool = False,
    output_dir: Optional[Path] = None,
) -> dict[str, SchedulerStats]:
    """Run all trials for all schedulers."""
    stats = {}

    # Run EEVDF (no sched_ext) trials first
    print("\n" + "=" * 70)
    print("Running EEVDF (native scheduler) trials")
    print("=" * 70)

    stop_all_scx_schedulers()
    eevdf_stats = SchedulerStats(scheduler="eevdf")

    for trial in range(1, n_trials + 1):
        result = run_single_trial(
            "eevdf", trial, schtest_path, None, duration, save_raw, output_dir
        )
        eevdf_stats.trials.append(result)

    stats["eevdf"] = eevdf_stats

    # Run trials for each LAVD version
    for name, lavd_path in lavd_binaries.items():
        print("\n" + "=" * 70)
        print(f"Running LAVD {name} trials")
        print(f"  Binary: {lavd_path}")
        print("=" * 70)

        sched_stats = SchedulerStats(scheduler=f"lavd_{name}")

        for trial in range(1, n_trials + 1):
            # Start the scheduler
            try:
                lavd_proc = start_lavd(lavd_path)
            except Exception as e:
                print(f"  ERROR: Failed to start LAVD: {e}")
                break

            try:
                result = run_single_trial(
                    f"lavd_{name}",
                    trial,
                    schtest_path,
                    lavd_path,
                    duration,
                    save_raw,
                    output_dir,
                )
                sched_stats.trials.append(result)
            finally:
                stop_lavd(lavd_proc)

            # Brief pause between trials
            time.sleep(1)

        stats[f"lavd_{name}"] = sched_stats

    return stats


def print_summary(stats: dict[str, SchedulerStats]):
    """Print summary statistics."""
    print("\n")
    print("=" * 80)
    print("SUMMARY: Victim-Control Gap Distribution")
    print("=" * 80)
    print()
    print("Gap = (control_sched_time - victim_sched_time) / control_sched_time * 100")
    print("Higher gap = more unfairness (victim got less time due to IRQ load)")
    print()
    print(
        f"{'Scheduler':<25} {'N':>4} {'Mean Gap':>10} {'Stdev':>8} {'Min':>8} {'Median':>8} {'Max':>8}"
    )
    print("-" * 80)

    for name, s in sorted(stats.items()):
        if not s.trials:
            print(f"{name:<25} {'(no data)':>10}")
            continue

        print(
            f"{s.scheduler:<25} {len(s.trials):>4} "
            f"{s.mean_gap:>9.2f}% {s.stdev_gap:>7.2f}% "
            f"{s.min_gap:>7.2f}% {s.median_gap:>7.2f}% {s.max_gap:>7.2f}%"
        )

    print("-" * 80)
    print()

    # Comparison analysis
    eevdf = stats.get("eevdf")
    lavd_main = stats.get("lavd_main")
    lavd_sans = stats.get("lavd_sans_irq_accounting")
    lavd_irq = stats.get("lavd_irq")

    if eevdf and lavd_main:
        print("Analysis:")
        print(f"  EEVDF baseline gap:     {eevdf.mean_gap:.2f}% ± {eevdf.stdev_gap:.2f}%")
        print(f"  LAVD main gap:          {lavd_main.mean_gap:.2f}% ± {lavd_main.stdev_gap:.2f}%")

        if lavd_sans:
            diff = lavd_main.mean_gap - lavd_sans.mean_gap
            print(
                f"  LAVD sans_irq gap:      {lavd_sans.mean_gap:.2f}% ± {lavd_sans.stdev_gap:.2f}%"
            )
            print(f"    Difference (main - sans_irq): {diff:+.2f}%")

        if lavd_irq:
            diff = lavd_main.mean_gap - lavd_irq.mean_gap
            print(
                f"  LAVD irq gap:           {lavd_irq.mean_gap:.2f}% ± {lavd_irq.stdev_gap:.2f}%"
            )
            print(f"    Difference (main - irq): {diff:+.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run IRQ accounting trials across LAVD versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trials", "-n", type=int, default=10, help="Number of trials per scheduler (default: 10)"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=10, help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building LAVD binaries (use existing ones in ./bin)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build binaries, don't run trials (no sudo needed)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory to save raw output files",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw test output to files",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=list(LAVD_VERSIONS.keys()) + ["all"],
        default=["all"],
        help="Which LAVD versions to test (default: all)",
    )
    parser.add_argument(
        "--scx-repo",
        type=Path,
        help="Path to existing scx repo (avoids cloning)",
    )

    args = parser.parse_args()

    # Check sched_ext support
    state, _ = get_sched_ext_state()
    if state == "not_supported":
        print("ERROR: sched_ext not supported on this kernel.")
        sys.exit(1)

    # Determine repo root
    repo_root = Path(__file__).parent.parent.resolve()
    print(f"Repository root: {repo_root}")

    # Set up output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif args.save_raw:
        args.output_dir = repo_root / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving raw output to: {args.output_dir}")

    # Determine which versions to test
    versions_to_test = (
        list(LAVD_VERSIONS.keys()) if "all" in args.versions else args.versions
    )

    # Build schtest
    schtest_path = build_schtest(repo_root)

    # Build or find LAVD binaries
    lavd_binaries = {}
    bin_dir = repo_root / BIN_DIR

    if args.skip_build:
        print("\nUsing existing binaries from ./bin")
        for name in versions_to_test:
            bin_path = bin_dir / f"scx_lavd_{name}"
            if bin_path.exists():
                lavd_binaries[name] = bin_path
                print(f"  Found: {bin_path}")
            else:
                print(f"  WARNING: Binary not found: {bin_path}")
    else:
        # Use provided scx repo or clone to temp directory
        if args.scx_repo:
            if not args.scx_repo.exists():
                print(f"ERROR: SCX repo not found: {args.scx_repo}")
                sys.exit(1)
            scx_dir = args.scx_repo.resolve()
            print(f"\nUsing existing SCX repo: {scx_dir}")
            work_dir = None
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="scx_build_"))
            print(f"\nBuild directory: {work_dir}")
            scx_dir = clone_scx_repo(work_dir)

        try:
            for name in versions_to_test:
                commit = LAVD_VERSIONS[name]
                output_name = f"scx_lavd_{name}"
                try:
                    lavd_binaries[name] = build_lavd_at_commit(
                        scx_dir, commit, output_name, bin_dir
                    )
                except Exception as e:
                    print(f"  WARNING: Failed to build {name}: {e}")
        finally:
            # Clean up build directory if we created one
            if work_dir:
                print(f"\nCleaning up build directory: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)

    if not lavd_binaries:
        print("\nWARNING: No LAVD binaries available. Will only test EEVDF.")

    # If build-only, exit now
    if args.build_only:
        print("\n--build-only specified, skipping trials.")
        print(f"Binaries are in: {bin_dir}")
        sys.exit(0)

    # Check we can use sudo for trials
    print("\nChecking sudo access for running trials...")
    try:
        subprocess.run(["sudo", "-n", "true"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("ERROR: Trials require sudo access. Please run with sudo or configure passwordless sudo.")
        print("Alternatively, use --build-only to just build the binaries.")
        sys.exit(1)

    # Run trials
    print(f"\n{'='*70}")
    print(f"Running {args.trials} trials per scheduler, {args.duration}s each")
    print(f"Schedulers: eevdf, " + ", ".join(f"lavd_{n}" for n in lavd_binaries.keys()))
    print(f"{'='*70}")

    stats = run_all_trials(
        schtest_path,
        lavd_binaries,
        args.trials,
        args.duration,
        args.save_raw,
        args.output_dir,
    )

    # Print summary
    print_summary(stats)

    # Save JSON if requested
    if args.json:
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "trials_per_scheduler": args.trials,
            "duration_seconds": args.duration,
            "lavd_versions": LAVD_VERSIONS,
            "results": {name: s.to_dict() for name, s in stats.items()},
        }
        args.json.write_text(json.dumps(json_data, indent=2))
        print(f"\nResults saved to: {args.json}")


if __name__ == "__main__":
    main()
