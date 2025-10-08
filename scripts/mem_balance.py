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
import time
import select
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, List, Any
from enum import Enum
from dataclasses import dataclass

# Configuration
EXPERIMENT_DURATION = 6  # seconds
RESULTS_DIR = Path("./results")
STRESS_SCRIPT = "./stress.sh"
SCX_DIR = Path("../../scx")

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
    SCX_LAVD = "scx_lavd"

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

class SchedStartMonitor:
    """Monitor dmesg for scheduler enabled messages."""

    def __init__(self, scheduler_name: str) -> None:
        """Start dmesg monitoring for the given scheduler."""
        self.scheduler_name = scheduler_name
        self.dmesg_proc = subprocess.Popen(
            ["sudo", "dmesg", "-W"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        if self.dmesg_proc.stdout is None:
            self.teardown()
            raise RuntimeError("Failed to start dmesg monitoring")

        print(f"Started dmesg monitoring for {scheduler_name} scheduler...")

    def wait_for_sched_enabled(self, timeout: float = 30.0) -> None:
        """Wait for scheduler enabled message."""
        print(f"Waiting for {self.scheduler_name} scheduler to be enabled...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.dmesg_proc.poll() is not None:
                # dmesg process died
                raise RuntimeError("dmesg monitoring process died")

            # Read with timeout to avoid blocking forever
            ready, _, _ = select.select([self.dmesg_proc.stdout], [], [], 1.0)

            if ready and self.dmesg_proc.stdout:
                line = self.dmesg_proc.stdout.readline()
                if line:
                    print(f"dmesg: {line.strip()}")  # Debug output
                    # Look for scheduler enabled message
                    if f'sched_ext: BPF scheduler "{self.scheduler_name}_' in line and "enabled" in line:
                        print(f"Scheduler {self.scheduler_name} enabled successfully")
                        time.sleep(1)  # Give it a moment to fully initialize
                        return

        raise RuntimeError(f"Timeout waiting for {self.scheduler_name} scheduler to be enabled")

    def teardown(self) -> None:
        """Clean up dmesg monitoring process."""
        if hasattr(self, 'dmesg_proc') and self.dmesg_proc:
            self.dmesg_proc.terminate()
            try:
                self.dmesg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dmesg_proc.kill()
                self.dmesg_proc.wait()

class ExperimentRunner:
    def __init__(self, machine_name: Optional[str] = None) -> None:
        self.num_cores = self._get_num_cores()
        self.machine_name = machine_name or self._get_machine_name_from_cpuinfo()
        self.results: List[ExperimentResult] = []
        # Create machine-specific results directory
        self.results_dir = RESULTS_DIR / self.machine_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_counter = 1

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

    def _start_scheduler(self, scheduler: SchedulerType) -> Optional[subprocess.Popen]:
        """Start a scheduler process and wait for it to be enabled."""
        if scheduler == SchedulerType.DEFAULT:
            return None  # Default scheduler is always active

        if scheduler == SchedulerType.SCX_LAVD:
            scheduler_path = SCX_DIR / "target/release/scx_lavd"
            if not scheduler_path.exists():
                raise FileNotFoundError(f"Scheduler not found: {scheduler_path}")

            print(f"Starting scheduler: {scheduler_path}")

            # Start dmesg monitoring first
            scheduler_name_map = {
                SchedulerType.SCX_LAVD: "lavd"
            }
            scheduler_name = scheduler_name_map[scheduler]

            monitor = SchedStartMonitor(scheduler_name)

            try:
                # Start the scheduler process with sudo
                proc = subprocess.Popen(
                    ["sudo", str(scheduler_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Wait for scheduler to be enabled
                monitor.wait_for_sched_enabled()
                return proc

            except Exception:
                # Clean up monitor if scheduler startup fails
                monitor.teardown()
                raise
            finally:
                # Always clean up the monitor
                monitor.teardown()

        raise ValueError(f"Unknown scheduler: {scheduler}")

    def _get_cpu_stress_params(self, num_cores: int) -> str:
        """Get stress-ng CPU workload parameters as a function of core count."""
        return f"--cpu {num_cores} --cpu-method int64"

    def _get_mem_stress_params(self, num_cores: int) -> str:
        """Get stress-ng memory workload parameters as a function of core count."""
        return f"--vm {num_cores} --vm-keep --vm-method ror --vm-bytes {num_cores}g"

    def _stop_scheduler(self, proc: Optional[subprocess.Popen]) -> None:
        """Stop a scheduler process."""
        if proc is None:
            return

        print("Stopping scheduler...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.CalledProcessError:
            print("Scheduler didn't terminate gracefully, killing...")
            proc.kill()
            proc.wait()
        print("Scheduler stopped")

    def _create_stress_script(self, workload: WorkloadType, pinning: PinningStrategy,
                             scheduler: SchedulerType, run_dir: Path) -> str:
        """Create stress.sh script for given configuration."""
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

        # Create script content based on workload with run-specific YAML files
        cpu_yaml = run_dir / "metrics_cpu.yaml"
        mem_yaml = run_dir / "metrics_mem.yaml"

        # Get abstracted stress-ng parameters
        cpu_params = self._get_cpu_stress_params(P)
        mem_params = self._get_mem_stress_params(P)

        if workload == WorkloadType.BOTH:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
(stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml {cpu_yaml} \\
   {cpu_params} {cpu_taskset}) &
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml {mem_yaml} \\
   {mem_params} {mem_taskset};
"""
        elif workload == WorkloadType.CPU:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml {cpu_yaml} \\
   {cpu_params} {cpu_taskset};
"""
        elif workload == WorkloadType.MEM:
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml {mem_yaml} \\
   {mem_params} {mem_taskset};
"""
        else:
            raise ValueError(f"Unknown workload: {workload}")

        return script_content

    def _run_experiment(self, workload: WorkloadType, pinning: PinningStrategy, scheduler: SchedulerType = SchedulerType.DEFAULT) -> ExperimentResult:
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"Running experiment: workload={workload.value}, pinning={pinning.value}, scheduler={scheduler.value}")
        print(f"{'='*60}")

        # Create run-specific directory
        full_config = self._get_full_config_name(workload, pinning, scheduler)
        run_name = f"run_{self.run_counter:03d}_{full_config}"
        run_dir = self.results_dir / run_name
        run_dir.mkdir(exist_ok=True)

        print(f"Run directory: {run_dir}")

        # Create and write stress script in run directory
        stress_script = run_dir / "stress.sh"
        script_content = self._create_stress_script(workload, pinning, scheduler, run_dir)
        with open(stress_script, 'w') as f:
            f.write(script_content)
        os.chmod(stress_script, 0o755)

        # Start scheduler if needed
        scheduler_proc = None
        try:
            scheduler_proc = self._start_scheduler(scheduler)

            # Run experiment with perf in run directory
            perf_output_file = run_dir / "perf.json"
            perf_cmd = [
                "perf", "stat", "-j",
                "-e", "instructions,cycles,cache-references,cache-misses",
                str(stress_script)
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

        finally:
            # Always stop the scheduler if we started one
            self._stop_scheduler(scheduler_proc)

        # Parse results
        cpu_metrics: Optional[StressMetrics] = None
        mem_metrics: Optional[StressMetrics] = None

        # Parse stress-ng YAML files from run directory
        if workload in [WorkloadType.BOTH, WorkloadType.CPU]:
            cpu_yaml = run_dir / "metrics_cpu.yaml"
            cpu_metrics = self._parse_stress_yaml(str(cpu_yaml))

        if workload in [WorkloadType.BOTH, WorkloadType.MEM]:
            mem_yaml = run_dir / "metrics_mem.yaml"
            mem_metrics = self._parse_stress_yaml(str(mem_yaml))

        # Parse perf JSON output
        perf_metrics = self._parse_perf_json(perf_output_file)

        # Create typed result
        experiment_result = ExperimentResult(
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

        # Increment run counter for next experiment
        self.run_counter += 1

        return experiment_result

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

        # Create/update latest symlink
        self._update_latest_symlink()

    def _update_latest_symlink(self) -> None:
        """Create or update symlink to latest results."""
        latest_path = RESULTS_DIR / "latest"

        # Remove existing symlink if it exists
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()

        # Create new symlink to current results directory
        latest_path.symlink_to(self.results_dir.name)
        print(f"Updated latest results symlink: {latest_path} -> {self.results_dir.name}")

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
        """Create improved stacked bar chart visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'CPU Scheduling Experiment Results\n'
                     f'100% CPU = {max_cpu_persec:,.0f} ops/sec, '
                     f'100% MEM = {max_mem_persec:,.0f} ops/sec')

        # Consistent colors for CPU and MEM across all bars
        cpu_color = '#2E86AB'  # Blue for CPU
        mem_color = '#A23B72'  # Purple for MEM

        # Plot: Stacked bar chart with consistent colors
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

            # Create stacked bars with consistent colors
            x_offset = x_pos + i * width
            cpu_bars = ax.bar(x_offset, cpu_values, width,
                            label='CPU' if i == 0 else '',
                            color=cpu_color, alpha=0.8)
            mem_bars = ax.bar(x_offset, mem_values, width, bottom=cpu_values,
                            label='MEM' if i == 0 else '',
                            color=mem_color, alpha=0.8)

            # Add percentage labels on bars
            for j, (cpu_bar, mem_bar, cpu_val, mem_val) in enumerate(zip(cpu_bars, mem_bars, cpu_values, mem_values)):
                # CPU label (middle of CPU portion)
                if cpu_val > 5:  # Only show label if bar is tall enough
                    ax.text(cpu_bar.get_x() + cpu_bar.get_width()/2, cpu_val/2,
                           f'{cpu_val:.0f}%', ha='center', va='center',
                           fontweight='bold', fontsize=9, color='white')

                # MEM label (middle of MEM portion)
                if mem_val > 5:  # Only show label if bar is tall enough
                    ax.text(mem_bar.get_x() + mem_bar.get_width()/2,
                           cpu_val + mem_val/2,
                           f'{mem_val:.0f}%', ha='center', va='center',
                           fontweight='bold', fontsize=9, color='white')

        # Update x-axis labels to show pinning strategies
        workload_labels = [f'{wl}\n(none/spread/half)' for wl in workloads]

        ax.set_xlabel('Workload Type (Pinning Strategies Left to Right)')
        ax.set_ylabel('Normalized Performance (%)')
        ax.set_title('CPU Scheduling Performance by Workload and Pinning Strategy')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(workload_labels)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 220)

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
