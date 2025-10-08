#!/usr/bin/env python3
"""
Quick test script to validate the approach with shorter duration
"""

import sys
import argparse
from typing import Optional
from pathlib import Path
sys.path.insert(0, '/home/newton/playground/schtest/scripts')

# Import our main script
from mem_balance import ExperimentRunner, WorkloadType, PinningStrategy, SchedulerType

# Create a test runner with shorter duration
class TestRunner(ExperimentRunner):
    def __init__(self, machine_name: Optional[str] = None) -> None:
        super().__init__(machine_name=machine_name)
        # No need to reference EXPERIMENT_DURATION since we override the script generation

    def _create_stress_script(self, workload: WorkloadType, pinning: PinningStrategy, 
                             scheduler: SchedulerType, run_dir: Path) -> str:
        """Override to use shorter duration."""
        P = self.num_cores
        run_config = self._get_run_config_name(pinning, scheduler)
        run_name = run_dir.name

        # Determine taskset arguments based on pinning strategy
        if pinning == PinningStrategy.NONE:
            cpu_taskset = ""
            mem_taskset = ""
        elif pinning == PinningStrategy.SPREAD:
            cpu_taskset = f"--taskset 0-{P-1}"
            mem_taskset = f"--taskset {P}-{2*P-1}"
        elif pinning == PinningStrategy.HALF:
            cpu_taskset = "--taskset even"
            mem_taskset = "--taskset odd"
        else:
            raise ValueError(f"Unknown pinning strategy: {pinning}")

        # Create script content based on workload (with 2 second duration)
        cpu_yaml = run_dir / "metrics_cpu.yaml"
        mem_yaml = run_dir / "metrics_mem.yaml"
        
        if workload == WorkloadType.BOTH:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
(stress-ng --metrics -t 2 --yaml {cpu_yaml} \\
   --cpu {P} --cpu-method int64 {cpu_taskset}) &
stress-ng --metrics -t 2 --yaml {mem_yaml} \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        elif workload == WorkloadType.CPU:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t 2 --yaml {cpu_yaml} \\
   --cpu {P} --cpu-method int64 {cpu_taskset};
"""
        elif workload == WorkloadType.MEM:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t 2 --yaml {mem_yaml} \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        else:
            raise ValueError(f"Unknown workload: {workload}")

        return script_content

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick CPU Scheduling Experiment Test")
    parser.add_argument("--machine", type=str, default=None,
                       help="Machine name tag (default: auto-detect from /proc/cpuinfo)")
    args = parser.parse_args()

    print("Running quick test with 2-second experiments...")
    runner = TestRunner(machine_name=args.machine)
    print(f"Machine: {runner.machine_name}")
    print(f"Results directory: {runner.results_dir}")

    # Test just one configuration
    result = runner._run_experiment(WorkloadType.CPU, PinningStrategy.SPREAD)
    print("\nTest result:")
    print(f"  workload: {result.workload.value}")
    print(f"  pinning: {result.pinning.value}")
    print(f"  scheduler: {result.scheduler.value}")
    print(f"  bogo_cpu: {result.bogo_cpu}")
    print(f"  bogo_cpu_persec: {result.bogo_cpu_persec}")
    print(f"  instructions: {result.instructions}")
    print(f"  cycles: {result.cycles}")

if __name__ == "__main__":
    main()