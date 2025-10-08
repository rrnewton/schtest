#!/usr/bin/env python3
"""
Quick test script to validate the approach with shorter duration
"""

import sys
sys.path.insert(0, '/home/newton/playground/schtest/scripts')

# Import our main script
from mem_balance import ExperimentRunner, WorkloadType, PinningStrategy

# Create a test runner with shorter duration
class TestRunner(ExperimentRunner):
    def __init__(self) -> None:
        super().__init__()
        # No need to reference EXPERIMENT_DURATION since we override the script generation

    def _create_stress_script(self, workload: WorkloadType, pinning: PinningStrategy) -> str:
        """Override to use shorter duration."""
        P = self.num_cores

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
        if workload == WorkloadType.BOTH:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
(stress-ng --metrics -t 2 --yaml metrics_cpu_{pinning.value}.yaml \\
   --cpu {P} --cpu-method int64 {cpu_taskset}) &
stress-ng --metrics -t 2 --yaml metrics_mem_{pinning.value}.yaml \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        elif workload == WorkloadType.CPU:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t 2 --yaml metrics_cpu_{pinning.value}.yaml \\
   --cpu {P} --cpu-method int64 {cpu_taskset};
"""
        elif workload == WorkloadType.MEM:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t 2 --yaml metrics_mem_{pinning.value}.yaml \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        else:
            raise ValueError(f"Unknown workload: {workload}")

        return script_content

def main() -> None:
    print("Running quick test with 2-second experiments...")
    runner = TestRunner()

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