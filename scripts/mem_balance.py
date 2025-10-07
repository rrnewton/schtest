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
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, Any

# Configuration
EXPERIMENT_DURATION = 10  # seconds
RESULTS_DIR = Path("./results")
STRESS_SCRIPT = "./stress.sh"

class ExperimentRunner:
    def __init__(self):
        self.num_cores = self._get_num_cores()
        self.results = []
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)

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

    def _create_stress_script(self, workload: str, pinning: str) -> str:
        """Create stress.sh script for given configuration."""
        P = self.num_cores

        # Determine taskset arguments based on pinning strategy
        if pinning == "none":
            cpu_taskset = ""
            mem_taskset = ""
        elif pinning == "spread":
            cpu_taskset = f"--taskset 0-{P-1}"
            mem_taskset = f"--taskset {P}-{2*P-1}"
        elif pinning == "half":
            cpu_taskset = "--taskset even"
            mem_taskset = "--taskset odd"
        else:
            raise ValueError(f"Unknown pinning strategy: {pinning}")

        # Create script content based on workload
        if workload == "both":
            script_content = f"""#!/bin/bash
set -xeuo pipefail
(stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_cpu_{pinning}.yaml \\
   --cpu {P} --cpu-method int64 {cpu_taskset}) &
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_mem_{pinning}.yaml \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        elif workload == "cpu":
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_cpu_{pinning}.yaml \\
   --cpu {P} --cpu-method int64 {cpu_taskset};
"""
        elif workload == "mem":
            script_content = f"""#!/bin/bash
set -xeuo pipefail
stress-ng --metrics -t {EXPERIMENT_DURATION} --yaml metrics_mem_{pinning}.yaml \\
   --vm {P} --vm-keep --vm-method ror --vm-bytes {P}g {mem_taskset};
"""
        else:
            raise ValueError(f"Unknown workload: {workload}")

        return script_content

    def _run_experiment(self, workload: str, pinning: str, scheduler: str = "default") -> Dict[str, Any]:
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"Running experiment: workload={workload}, pinning={pinning}, scheduler={scheduler}")
        print(f"{'='*60}")

        # Create and write stress script
        script_content = self._create_stress_script(workload, pinning)
        with open(STRESS_SCRIPT, 'w') as f:
            f.write(script_content)
        os.chmod(STRESS_SCRIPT, 0o755)

        # Run experiment with perf
        perf_output_file = self.results_dir / f"perf_{workload}_{pinning}_{scheduler}.json"
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
        experiment_result = {
            'workload': workload,
            'pinning': pinning,
            'scheduler': scheduler,
            'num_cores': self.num_cores
        }

        # Parse stress-ng YAML files
        if workload in ["both", "cpu"]:
            cpu_metrics = self._parse_stress_yaml(f"metrics_cpu_{pinning}.yaml")
            if cpu_metrics:
                experiment_result.update({
                    'bogo_cpu': cpu_metrics.get('bogo_ops', 0),
                    'bogo_cpu_persec': cpu_metrics.get('bogo_ops_per_sec_cpu_time', 0),
                    'real_time_cpu': cpu_metrics.get('real_time', 0),
                })

        if workload in ["both", "mem"]:
            mem_metrics = self._parse_stress_yaml(f"metrics_mem_{pinning}.yaml")
            if mem_metrics:
                experiment_result.update({
                    'bogo_mem': mem_metrics.get('bogo_ops', 0),
                    'bogo_mem_persec': mem_metrics.get('bogo_ops_per_sec_cpu_time', 0),
                    'real_time_mem': mem_metrics.get('real_time', 0),
                })

        # Parse perf JSON output
        perf_metrics = self._parse_perf_json(perf_output_file)
        experiment_result.update(perf_metrics)

        # Fill missing values with 0
        for key in ['bogo_cpu', 'bogo_cpu_persec', 'bogo_mem', 'bogo_mem_persec',
                   'real_time_cpu', 'real_time_mem', 'instructions', 'cycles',
                   'cache_refs', 'cache_misses']:
            if key not in experiment_result:
                experiment_result[key] = 0

        return experiment_result

    def _parse_stress_yaml(self, yaml_file: str) -> Optional[Dict[str, Any]]:
        """Parse stress-ng YAML output file."""
        if not os.path.exists(yaml_file):
            print(f"Warning: {yaml_file} not found")
            return None

        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            if not data or 'metrics' not in data:
                print(f"Warning: No metrics found in {yaml_file}")
                return None

            metrics = data['metrics'][0]  # Assuming first metric
            return {
                'bogo_ops': metrics.get('bogo-ops', 0),
                'bogo_ops_per_sec_cpu_time': metrics.get('bogo-ops-per-second-usr-sys-time', 0),
                'real_time': metrics.get('bogo-ops-per-second-real-time', 0),
            }
        except Exception as e:
            print(f"Error parsing {yaml_file}: {e}")
            return None

    def _parse_perf_json(self, json_file: Path) -> Dict[str, float]:
        """Parse perf stat JSON output."""
        perf_metrics: Dict[str, float] = {
            'instructions': 0.0,
            'cycles': 0.0,
            'cache_refs': 0.0,
            'cache_misses': 0.0
        }

        if not json_file.exists():
            print(f"Warning: {json_file} not found")
            return perf_metrics

        try:
            with open(json_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"Warning: {json_file} is empty")
                    return perf_metrics

                # Parse line by line since perf outputs one JSON object per line
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
                                perf_metrics['instructions'] = value
                            elif event == 'cycles':
                                perf_metrics['cycles'] = value
                            elif event == 'cache-references':
                                perf_metrics['cache_refs'] = value
                            elif event == 'cache-misses':
                                perf_metrics['cache_misses'] = value
                    except json.JSONDecodeError:
                        continue  # Skip non-JSON lines

        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

        return perf_metrics

    def run_all_experiments(self):
        """Run all experiment combinations."""
        workloads = ["both", "cpu", "mem"]
        pinning_strategies = ["none", "spread", "half"]

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

    def _save_results(self):
        """Save results to CSV file."""
        df = pd.DataFrame(self.results)
        csv_file = self.results_dir / "experiment_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

    def analyze_and_plot(self):
        """Analyze results and create visualization."""
        if not self.results:
            print("No results to analyze!")
            return

        df = pd.DataFrame(self.results)

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

    def _create_plots(self, df: pd.DataFrame, max_cpu_persec: float, max_mem_persec: float):
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

            cpu_values = []
            mem_values = []

            for workload in workloads:
                row = subset[subset['workload'] == workload]
                if len(row) > 0:
                    cpu_values.append(row['cpu_normalized'].iloc[0])
                    mem_values.append(row['mem_normalized'].iloc[0])
                else:
                    cpu_values.append(0)
                    mem_values.append(0)

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
        combined_data = []
        labels = []
        colors = []

        color_map = {'none': 'red', 'spread': 'green', 'half': 'blue'}

        for workload in workloads:
            workload_data = []
            for pinning in pinning_strategies:
                subset = df[(df['workload'] == workload) & (df['pinning'] == pinning)]
                if len(subset) > 0:
                    workload_data.append(subset['combined_tput'].iloc[0])
                    labels.append(f'{workload}\n{pinning}')
                    colors.append(color_map[pinning])
                else:
                    workload_data.append(0)
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

def main():
    """Main experiment execution."""
    print("CPU Scheduling Experiment")
    print("=" * 40)

    runner = ExperimentRunner()
    print(f"Detected {runner.num_cores} physical cores")

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
