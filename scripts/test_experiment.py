#!/usr/bin/env python3
"""
Quick test script to validate the approach with shorter duration (2 seconds)

This is used by `make quick_test` and `make validate` to ensure basic functionality works.
"""

import sys
import argparse
from typing import Optional
sys.path.insert(0, '/home/newton/playground/schtest/scripts')

# Import our main script
from mem_balance import ExperimentRunner, WorkloadType, PinningStrategy, StressorType
from sched_monitor import SchedulerType

# Temporarily patch EXPERIMENT_DURATION for testing
import mem_balance
mem_balance.EXPERIMENT_DURATION = 2

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick CPU Scheduling Experiment Test")
    parser.add_argument("--machine", type=str, default=None,
                       help="Machine name tag (default: auto-detect from /proc/cpuinfo)")
    parser.add_argument("--scheduler", type=str, default="default",
                       choices=["default", "scx_lavd"],
                       help="Scheduler type to test (default: default)")
    args = parser.parse_args()

    print("Running quick test with 2-second experiments...")

    # Test specified scheduler
    scheduler = SchedulerType.DEFAULT if args.scheduler == "default" else SchedulerType.SCX_LAVD

    # Create runner with single CPU/SPREAD test configuration
    runner = ExperimentRunner(
        machine_name=args.machine,
        workloads=[WorkloadType.CPU],
        pinning_strategies=[PinningStrategy.SPREAD],
        schedulers=[scheduler],
        stressor=StressorType.STRESS_NG,  # Use stress-ng for quick test (faster than rt-app)
        append_mode=False,
        trials=1
    )
    print(f"Machine: {runner.machine_name}")
    print(f"Results directory: {runner.results_dir}")

    result = runner._run_experiment(WorkloadType.CPU, PinningStrategy.SPREAD, scheduler, StressorType.STRESS_NG)
    print("\nTest result:")
    print(f"  workload: {result.params.workload.value}")
    print(f"  pinning: {result.params.pinning.value}")
    print(f"  scheduler: {result.params.scheduler.value}")
    print(f"  machine: {result.params.machine}")
    print(f"  kernel: {result.params.kernel}")
    print(f"  bogo_cpu: {result.bogo_cpu}")
    print(f"  bogo_cpu_persec: {result.bogo_cpu_persec}")
    print(f"  instructions: {result.instructions}")
    print(f"  cycles: {result.cycles}")

    # Update latest symlink for quick testing
    runner._update_latest_symlink()

if __name__ == "__main__":
    main()