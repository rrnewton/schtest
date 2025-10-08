#!/usr/bin/env python3
"""
CPU Scheduling Experiment Script
Runs stress-ng workloads with different thread pinning strategies
and analyzes performance metrics.
"""

import os
import subprocess
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, List, Any
from enum import Enum
from dataclasses import dataclass

# Configuration
EXPERIMENT_DURATION = 10  # seconds
RESULTS_DIR = Path("./results")
STRESS_SCRIPT = "./stress.sh"

# Type Definitions
class WorkloadType(Enum):
    """Enum for workload types"""
    BOTH = "both"
    CPU = "cpu"
    MEM = "mem"

class PinningStrategy(Enum):
    """Enum for thread pinning strategies"""
    NONE = "none"
    SPREAD = "spread"
    HALF = "half"

class SchedulerType(Enum):
    """Enum for scheduler types"""
    DEFAULT = "default"

@dataclass
class StressMetrics:
    """Typed structure for stress-ng metrics"""
    bogo_ops: int
    bogo_ops_per_sec_cpu_time: float
    real_time: float

@dataclass
class PerfMetrics:
    """Typed structure for perf metrics"""
    instructions: float
    cycles: float
    cache_refs: float
    cache_misses: float

@dataclass
class ExperimentResult:
    """Typed structure for complete experiment results"""
    workload: WorkloadType
    pinning: PinningStrategy
    scheduler: SchedulerType
    num_cores: int
    bogo_cpu: int
    bogo_cpu_persec: float
    real_time_cpu: float
    bogo_mem: int
    bogo_mem_persec: float
    real_time_mem: float
    instructions: float
    cycles: float
    cache_refs: float
    cache_misses: float

class ExperimentRunner:
    def __init__(self, machine_name: Optional[str] = None) -> None:
        self.num_cores = self._get_num_cores()
        self.machine_name = machine_name or self._get_machine_name_from_cpuinfo()
        self.results: List[ExperimentResult] = []
        # Create machine-specific results directory
        self.results_dir = RESULTS_DIR / self.machine_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_num_cores(self) -> int:
        """Get number of physical cores on the system."""
        result = subprocess.run(
            ["lscpu", "-p=Core,Socket"],
            capture_output=True, text=True, check=True
        )
        lines = [line for line in result.stdout.split('\n') if not line.startswith('#') and line.strip()]
        unique_cores: Set[Tuple[str, str]] = set()
        for line in lines:
            if line.strip():
                core, socket = line.split(',')
                unique_cores.add((core, socket))
        return len(unique_cores)

    def _get_machine_name_from_cpuinfo(self) -> str:
        """Get machine name from /proc/cpuinfo model name."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        # Extract the model name after the colon
                        model_name = line.split(':', 1)[1].strip()
                        # Replace spaces and special characters with underscores
                        machine_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                        # Remove multiple consecutive underscores
                        machine_name = re.sub(r'_+', '_', machine_name)
                        # Remove leading/trailing underscores
                        return machine_name.strip('_')
        except (FileNotFoundError, IOError, IndexError):
            pass
        return "unknown_machine"

    def _get_run_config_name(self, pinning: PinningStrategy, scheduler: SchedulerType) -> str:
        """Generate run configuration name: pinning_scheduler (HOW we run)"""
        return f"{pinning.value}_{scheduler.value}"

    def _get_full_config_name(self, workload: WorkloadType, pinning: PinningStrategy, 
                             scheduler: SchedulerType) -> str:
        """Generate full configuration name: workload_pinning_scheduler (WHAT + HOW we run)"""
        run_config = self._get_run_config_name(pinning, scheduler)
        return f"{workload.value}_{run_config}"

    def _create_stress_script(self, workload: WorkloadType, pinning: PinningStrategy, 
                             scheduler: SchedulerType) -> str:
        """Create stress.sh script for given configuration."""
        P = self.num_cores
        run_config = self._get_run_config_name(pinning, scheduler)

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

        # Create script content based on workload
        if workload == WorkloadType.BOTH:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
(stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_cpu_{run_config}.yaml \\
   --cpu {P} --cpu-method int64 {cpu_taskset}) &
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_mem_{run_config}.yaml \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        elif workload == WorkloadType.CPU:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_cpu_{run_config}.yaml \\
   --cpu {P} --cpu-method int64 {cpu_taskset};
"""
        elif workload == WorkloadType.MEM:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_mem_{run_config}.yaml \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        else:
            raise ValueError(f"Unknown workload: {workload}")

        return script_content

    def _run_experiment(self, workload: WorkloadType, pinning: PinningStrategy, scheduler: SchedulerType = SchedulerType.DEFAULT) -> ExperimentResult:
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"Running experiment: workload={workload.value}, pinning={pinning.value}, scheduler={scheduler.value}")
        print(f"{'='*60}")

        # Create and write stress script
        script_content = self._create_stress_script(workload, pinning, scheduler)
        with open(STRESS_SCRIPT, 'w') as f:
            f.write(script_content)
        os.chmod(STRESS_SCRIPT, 0o755)

        # Run experiment with perf
        full_config = self._get_full_config_name(workload, pinning, scheduler)
        perf_output_file = self.results_dir / f"perf_{full_config}.json"
        perf_cmd = [
            "perf", "stat", "-j",
            "-e", "instructions,cycles,cache-references,cache-misses",
            STRESS_SCRIPT
        ]

        print(f"Running: {' '.join(perf_cmd)}")

        # Run the experiment
        with open(perf_output_file, 'w') as perf_file:
            result = subprocess.run(
                perf_cmd,
                stdout=subprocess.PIPE,
                stderr=perf_file,
                text=True
            )

        if result.returncode != 0:
            print(f"Warning: Experiment failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")

        # Parse results
        cpu_metrics: Optional[StressMetrics] = None
        mem_metrics: Optional[StressMetrics] = None
        run_config = self._get_run_config_name(pinning, scheduler)

        # Parse stress-ng YAML files
        if workload in [WorkloadType.BOTH, WorkloadType.CPU]:
            cpu_metrics = self._parse_stress_yaml(f"metrics_cpu_{run_config}.yaml")

        if workload in [WorkloadType.BOTH, WorkloadType.MEM]:
            mem_metrics = self._parse_stress_yaml(f"metrics_mem_{run_config}.yaml")

        # Parse perf JSON output
        perf_metrics = self._parse_perf_json(perf_output_file)

        # Create typed result
        return ExperimentResult(
            workload=workload,
            pinning=pinning,
            scheduler=scheduler,
            num_cores=self.num_cores,
            bogo_cpu=cpu_metrics.bogo_ops if cpu_metrics else 0,
            bogo_cpu_persec=cpu_metrics.bogo_ops_per_sec_cpu_time if cpu_metrics else 0.0,
            real_time_cpu=cpu_metrics.real_time if cpu_metrics else 0.0,
            bogo_mem=mem_metrics.bogo_ops if mem_metrics else 0,
            bogo_mem_persec=mem_metrics.bogo_ops_per_sec_cpu_time if mem_metrics else 0.0,
            real_time_mem=mem_metrics.real_time if mem_metrics else 0.0,
            instructions=perf_metrics.instructions,
            cycles=perf_metrics.cycles,
            cache_refs=perf_metrics.cache_refs,
            cache_misses=perf_metrics.cache_misses,
        )

    def _parse_stress_yaml(self, yaml_file: str) -> Optional[StressMetrics]:
        """Parse stress-ng YAML output file."""
        if not os.path.exists(yaml_file):
            print(f"YAML file not found: {yaml_file}")
            return None

        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            metrics = data.get('metrics', [{}])[0]
            return StressMetrics(
                bogo_ops=metrics.get('bogo-ops', 0),
                bogo_ops_per_sec_cpu_time=metrics.get('bogo-ops-per-second-usr-sys-time', 0.0),
                real_time=metrics.get('wall-clock-time', 0.0)
            )
        except Exception as e:
            print(f"Error parsing YAML file {yaml_file}: {e}")
            return None

    def _parse_perf_json(self, json_file: Path) -> PerfMetrics:
        """Parse perf stat JSON output."""
        default_metrics = PerfMetrics(
            instructions=0,
            cycles=0,
            cache_refs=0,
            cache_misses=0
        )

        if not json_file.exists():
            print(f"Warning: {json_file} not found")
            return default_metrics

        try:
            with open(json_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"Warning: {json_file} is empty")
                    return default_metrics

                # Parse line by line since perf outputs one JSON object per line
                instructions = 0
                cycles = 0
                cache_refs = 0
                cache_misses = 0

                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'counter-value' in data and 'event' in data:
                            event = data['event']
                            value = data['counter-value']

                            if event == 'instructions':
                                instructions = value
                            elif event == 'cycles':
                                cycles = value
                            elif event == 'cache-references':
                                cache_refs = value
                            elif event == 'cache-misses':
                                cache_misses = value
                    except json.JSONDecodeError:
                        continue  # Skip non-JSON lines

                return PerfMetrics(
                    instructions=instructions,
                    cycles=cycles,
                    cache_refs=cache_refs,
                    cache_misses=cache_misses
                )

        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            return default_metrics

    def run_all_experiments(self) -> None:
        """Run all experiment combinations."""
        workloads = [WorkloadType.BOTH, WorkloadType.CPU, WorkloadType.MEM]
        pinning_strategies = [PinningStrategy.NONE, PinningStrategy.SPREAD, PinningStrategy.HALF]

        total_experiments = len(workloads) * len(pinning_strategies)
        current_exp = 0

        for workload in workloads:
            for pinning in pinning_strategies:
                current_exp += 1
                print(f"\nProgress: {current_exp}/{total_experiments}")

                result = self._run_experiment(workload, pinning)
                self.results.append(result)

                # Save intermediate results
                self._save_results()

        print(f"\n{'='*60}")
        print("All experiments completed!")
        print(f"{'='*60}")

    def _save_results(self) -> None:
        """Save results to CSV file."""
        # Convert dataclasses to dictionaries with string enum values for CSV
        results_dicts: List[Dict[str, Any]] = []
        for result in self.results:
            result_dict: Dict[str, Any] = {
                'workload': result.workload.value,
                'pinning': result.pinning.value,
                'scheduler': result.scheduler.value,
                'num_cores': result.num_cores,
                'bogo_cpu': result.bogo_cpu,
                'bogo_cpu_persec': result.bogo_cpu_persec,
                'real_time_cpu': result.real_time_cpu,
                'bogo_mem': result.bogo_mem,
                'bogo_mem_persec': result.bogo_mem_persec,
                'real_time_mem': result.real_time_mem,
                'instructions': result.instructions,
                'cycles': result.cycles,
                'cache_refs': result.cache_refs,
                'cache_misses': result.cache_misses
            }
            results_dicts.append(result_dict)

        df = pd.DataFrame(results_dicts)
        csv_file = self.results_dir / "experiment_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

    def analyze_and_plot(self) -> None:
        """Analyze results and create visualization."""
        if not self.results:
            print("No results to analyze!")
            return

        # Convert dataclasses to dictionaries for analysis
        results_dicts: List[Dict[str, Any]] = []
        for result in self.results:
            result_dict: Dict[str, Any] = {
                'workload': result.workload.value,
                'pinning': result.pinning.value,
                'scheduler': result.scheduler.value,
                'num_cores': result.num_cores,
                'bogo_cpu': result.bogo_cpu,
                'bogo_cpu_persec': result.bogo_cpu_persec,
                'real_time_cpu': result.real_time_cpu,
                'bogo_mem': result.bogo_mem,
                'bogo_mem_persec': result.bogo_mem_persec,
                'real_time_mem': result.real_time_mem,
                'instructions': result.instructions,
                'cycles': result.cycles,
                'cache_refs': result.cache_refs,
                'cache_misses': result.cache_misses
            }
            results_dicts.append(result_dict)

        df = pd.DataFrame(results_dicts)

        # Calculate normalization factors (best performance = 100%)
        max_cpu_persec = df['bogo_cpu_persec'].max()
        max_mem_persec = df['bogo_mem_persec'].max()

        print(f"\nNormalization factors:")
        print(f"100% bogo CPU = {max_cpu_persec:,.0f} ops/sec")
        print(f"100% bogo MEM = {max_mem_persec:,.0f} ops/sec")

        # Normalize performance metrics
        df['cpu_normalized'] = (df['bogo_cpu_persec'] / max_cpu_persec * 100) if max_cpu_persec > 0 else 0
        df['mem_normalized'] = (df['bogo_mem_persec'] / max_mem_persec * 100) if max_mem_persec > 0 else 0
        df['combined_tput'] = df['cpu_normalized'] + df['mem_normalized']

        # Create visualization
        self._create_plots(df, max_cpu_persec, max_mem_persec)

        # Print summary table
        print("\nSummary Results:")
        summary_cols = ['workload', 'pinning', 'cpu_normalized', 'mem_normalized', 'combined_tput']
        print(df[summary_cols].round(1).to_string(index=False))

    def _create_plots(self, df: pd.DataFrame, max_cpu_persec: float, max_mem_persec: float) -> None:
        """Create stacked bar chart visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'CPU Scheduling Experiment Results\n'
                     f'100% CPU = {max_cpu_persec:,.0f} ops/sec, '
                     f'100% MEM = {max_mem_persec:,.0f} ops/sec')

        # Plot 1: Stacked bar chart
        workloads = ['both', 'cpu', 'mem']
        pinning_strategies = ['none', 'spread', 'half']

        x_pos = np.arange(len(workloads))
        width = 0.25

        for i, pinning in enumerate(pinning_strategies):
            subset = df[df['pinning'] == pinning]

            cpu_values: List[float] = []
            mem_values: List[float] = []

            for workload in workloads:
                row = subset[subset['workload'] == workload]
                if len(row) > 0:
                    cpu_values.append(float(row['cpu_normalized'].iloc[0]))
                    mem_values.append(float(row['mem_normalized'].iloc[0]))
                else:
                    cpu_values.append(0.0)
                    mem_values.append(0.0)

            # Create stacked bars
            x_offset = x_pos + i * width
            ax1.bar(x_offset, cpu_values, width, label=f'CPU ({pinning})',
                          alpha=0.8, color=f'C{i}')
            ax1.bar(x_offset, mem_values, width, bottom=cpu_values,
                          label=f'MEM ({pinning})', alpha=0.6, color=f'C{i+3}')

        ax1.set_xlabel('Workload Type')
        ax1.set_ylabel('Normalized Performance (%)')
        ax1.set_title(f'CPU Scheduling Experiment Results\n'
                     f'100% CPU = {max_cpu_persec:,.0f} ops/sec, '
                     f'100% MEM = {max_mem_persec:,.0f} ops/sec')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels(workloads)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 220)

        # Plot 2: Combined throughput comparison
        combined_data: List[float] = []
        labels: List[str] = []
        colors: List[str] = []

        color_map = {'none': 'red', 'spread': 'green', 'half': 'blue'}

        for workload in workloads:
            workload_data: List[float] = []
            for pinning in pinning_strategies:
                subset = df[(df['workload'] == workload) & (df['pinning'] == pinning)]
                if len(subset) > 0:
                    workload_data.append(float(subset['combined_tput'].iloc[0]))
                    labels.append(f'{workload}\n{pinning}')
                    colors.append(color_map[pinning])
                else:
                    workload_data.append(0.0)
            combined_data.extend(workload_data)

        x_pos2 = np.arange(len(labels))
        bars = ax2.bar(x_pos2, combined_data, color=colors, alpha=0.7)

        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Combined Throughput (%)')
        ax2.set_title('Combined CPU + Memory Throughput')
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 220)

        # Add value labels on bars
        for bar, value in zip(bars, combined_data):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save plot
        plot_file = self.results_dir / "experiment_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
        plt.show()

def main() -> None:
    """Main experiment execution."""
    parser = argparse.ArgumentParser(description="CPU Scheduling Experiment")
    parser.add_argument("--machine", type=str, default=None,
                       help="Machine name tag (default: auto-detect from /proc/cpuinfo)")
    args = parser.parse_args()

    print("CPU Scheduling Experiment")
    print("=" * 40)

    runner = ExperimentRunner(machine_name=args.machine)
    print(f"Machine: {runner.machine_name}")
    print(f"Detected {runner.num_cores} physical cores")
    print(f"Results directory: {runner.results_dir}")

    # Check dependencies
    try:
        subprocess.run(["stress-ng", "--version"],
                      capture_output=True, check=True)
        print("stress-ng is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: stress-ng is not installed!")
        print("Please install it with: sudo apt-get install stress-ng")
        return

    try:
        subprocess.run(["perf", "--version"],
                      capture_output=True, check=True)
        print("perf is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: perf is not installed!")
        print("Please install it with: sudo apt-get install linux-tools-generic")
        return

    # Run experiments
    runner.run_all_experiments()

    # Analyze and plot results
    runner.analyze_and_plot()

if __name__ == "__main__":
    main()
