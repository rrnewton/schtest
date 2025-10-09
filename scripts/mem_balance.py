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

# Import topology parser
from parse_topo import parse_topology, Machine

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
class ExperimentParams:
    """Parameters that uniquely identify an experiment run."""
    workload: WorkloadType
    scheduler: SchedulerType
    pinning: PinningStrategy
    num_cores: int
    experiment_duration: int
    machine: str
    kernel: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            'workload': self.workload.value,
            'scheduler': self.scheduler.value,
            'pinning': self.pinning.value,
            'num_cores': self.num_cores,
            'experiment_duration': self.experiment_duration,
            'machine': self.machine,
            'kernel': self.kernel
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentParams':
        """Create from dictionary loaded from YAML."""
        return cls(
            workload=WorkloadType(data['workload']),
            scheduler=SchedulerType(data['scheduler']),
            pinning=PinningStrategy(data['pinning']),
            num_cores=data['num_cores'],
            experiment_duration=data['experiment_duration'],
            machine=data['machine'],
            kernel=data['kernel']
        )

    def __eq__(self, other: object) -> bool:
        """Check if two experiment params are equivalent."""
        if not isinstance(other, ExperimentParams):
            return False
        return (
            self.workload == other.workload and
            self.scheduler == other.scheduler and
            self.pinning == other.pinning and
            self.num_cores == other.num_cores and
            self.experiment_duration == other.experiment_duration and
            self.machine == other.machine and
            self.kernel == other.kernel
        )

    def __hash__(self) -> int:
        """Make hashable for use in sets."""
        return hash((
            self.workload.value,
            self.scheduler.value,
            self.pinning.value,
            self.num_cores,
            self.experiment_duration,
            self.machine,
            self.kernel
        ))

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
    params: ExperimentParams
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV serialization."""
        result = self.params.to_dict()
        result.update({
            'bogo_cpu': self.bogo_cpu,
            'bogo_cpu_persec': self.bogo_cpu_persec,
            'real_time_cpu': self.real_time_cpu,
            'bogo_mem': self.bogo_mem,
            'bogo_mem_persec': self.bogo_mem_persec,
            'real_time_mem': self.real_time_mem,
            'instructions': self.instructions,
            'cycles': self.cycles,
            'cache_refs': self.cache_refs,
            'cache_misses': self.cache_misses
        })
        return result

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

        # Parse CPU topology
        print("Parsing CPU topology...")
        self.topology = parse_topology()
        print(f"Topology parsed: {len(self.topology.packages)} packages, {len(self.topology.get_cpu_numbers())} CPUs")

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

    def _start_scheduler(self, scheduler: SchedulerType) -> Optional[subprocess.Popen[str]]:
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

    def _stop_scheduler(self, proc: Optional[subprocess.Popen[str]]) -> None:
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

        # Determine taskset arguments based on pinning strategy using topology
        if pinning == PinningStrategy.NONE:
            cpu_taskset = ""
            mem_taskset = ""
        elif pinning == PinningStrategy.SPREAD:
            # Use split_hyperthreads to spread across all cores
            try:
                cpu_list, mem_list = self.topology.split_hyperthreads()
                cpu_taskset = f"--taskset {','.join(map(str, cpu_list))}"
                mem_taskset = f"--taskset {','.join(map(str, mem_list))}"
                print(f"SPREAD strategy: CPU cores {cpu_list}, MEM cores {mem_list}")
            except ValueError as e:
                print(f"Warning: Could not split hyperthreads ({e}), falling back to no pinning")
                cpu_taskset = ""
                mem_taskset = ""
        elif pinning == PinningStrategy.HALF:
            # Use split_dies to occupy only half of the cores physically
            try:
                cpu_list, mem_list = self.topology.split_dies()
                cpu_taskset = f"--taskset {','.join(map(str, cpu_list))}"
                mem_taskset = f"--taskset {','.join(map(str, mem_list))}"
                print(f"HALF strategy: CPU cores {cpu_list}, MEM cores {mem_list}")
            except ValueError as e:
                print(f"Warning: Could not split dies ({e}), falling back to no pinning")
                cpu_taskset = ""
                mem_taskset = ""
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

    def _get_kernel_version(self) -> str:
        """Get kernel version from uname -r."""
        try:
            result = subprocess.run(["uname", "-r"], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _create_experiment_params(self, workload: WorkloadType, pinning: PinningStrategy, scheduler: SchedulerType) -> ExperimentParams:
        """Create experiment parameters for this run."""
        return ExperimentParams(
            workload=workload,
            scheduler=scheduler,
            pinning=pinning,
            num_cores=self.num_cores,
            experiment_duration=EXPERIMENT_DURATION,
            machine=self.machine_name,
            kernel=self._get_kernel_version()
        )

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

        # Create experiment parameters
        params = self._create_experiment_params(workload, pinning, scheduler)

        # Create typed result
        experiment_result = ExperimentResult(
            params=params,
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

        # Save params.yaml to mark run as complete
        params_file = run_dir / "params.yaml"
        with open(params_file, 'w') as f:
            yaml.dump(params.to_dict(), f, default_flow_style=False)
        print(f"Saved params to {params_file}")

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

    def _load_params_from_run(self, run_dir: Path) -> Optional[ExperimentParams]:
        """Load experiment parameters from a run directory."""
        params_file = run_dir / "params.yaml"
        if not params_file.exists():
            return None

        try:
            with open(params_file, 'r') as f:
                data = yaml.safe_load(f)
            return ExperimentParams.from_dict(data)
        except Exception as e:
            print(f"Error loading params from {params_file}: {e}")
            return None

    def _load_existing_runs(self) -> List[ExperimentParams]:
        """Load all valid existing runs from the machine results directory."""
        existing_runs: List[ExperimentParams] = []

        if not self.results_dir.exists():
            return existing_runs

        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                params = self._load_params_from_run(run_dir)
                if params:
                    existing_runs.append(params)

        return existing_runs

    def _clean_incomplete_runs(self) -> Tuple[int, int]:
        """Clean up incomplete runs and return (complete_count, cleaned_count)."""
        import re
        
        complete_count = 0
        cleaned_count = 0
        max_run_counter = 0

        if not self.results_dir.exists():
            return complete_count, cleaned_count

        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                # Extract run counter from directory name (e.g., run_001_config -> 1)
                match = re.match(r'run_(\d+)_', run_dir.name)
                if match:
                    run_counter = int(match.group(1))
                    max_run_counter = max(max_run_counter, run_counter)
                
                params_file = run_dir / "params.yaml"

                if params_file.exists():
                    # Try to load params to validate completeness
                    params = self._load_params_from_run(run_dir)
                    if params:
                        complete_count += 1
                    else:
                        # Corrupt params file - clean up
                        print(f"Cleaning up run with corrupt params: {run_dir}")
                        subprocess.run(["rm", "-rf", str(run_dir)], check=True)
                        cleaned_count += 1
                else:
                    # No params file - incomplete run
                    print(f"Cleaning up incomplete run: {run_dir}")
                    subprocess.run(["rm", "-rf", str(run_dir)], check=True)
                    cleaned_count += 1

        # Initialize run counter to avoid duplicates
        self.run_counter = max_run_counter + 1
        if max_run_counter > 0:
            print(f"Initialized run counter to {self.run_counter} (max observed: {max_run_counter})")

        return complete_count, cleaned_count

    def _get_planned_experiments(self) -> List[ExperimentParams]:
        """Get list of all experiments we plan to run."""
        workloads = [WorkloadType.BOTH, WorkloadType.CPU, WorkloadType.MEM]
        pinning_strategies = [PinningStrategy.NONE, PinningStrategy.SPREAD, PinningStrategy.HALF]
        schedulers = [SchedulerType.DEFAULT, SchedulerType.SCX_LAVD]

        planned = []
        for scheduler in schedulers:
            for workload in workloads:
                for pinning in pinning_strategies:
                    params = self._create_experiment_params(workload, pinning, scheduler)
                    planned.append(params)

        return planned

    def _get_missing_experiments(self, existing: List[ExperimentParams], planned: List[ExperimentParams]) -> List[ExperimentParams]:
        """Get list of experiments that still need to be run."""
        existing_set = set(existing)
        return [params for params in planned if params not in existing_set]

    def run_all_experiments(self) -> None:
        """Run all experiment combinations with incremental support."""
        print("=" * 60)
        print("EXPERIMENTAL RUN PREPARATION")
        print("=" * 60)

        # Step 1: Clean up incomplete runs
        print("Cleaning up incomplete runs...")
        complete_count, cleaned_count = self._clean_incomplete_runs()
        print(f"Found {complete_count} complete runs, cleaned up {cleaned_count} incomplete runs")

        # Step 2: Load existing valid runs
        print("Loading existing completed runs...")
        existing_runs = self._load_existing_runs()
        print(f"Loaded {len(existing_runs)} existing runs")

        # Step 3: Determine what experiments we plan to run
        planned_experiments = self._get_planned_experiments()
        print(f"Total planned experiments: {len(planned_experiments)}")

        # Step 4: Find missing experiments
        missing_experiments = self._get_missing_experiments(existing_runs, planned_experiments)
        print(f"Missing experiments to run: {len(missing_experiments)}")

        if not missing_experiments:
            print("All experiments already completed!")
        else:
            print("\nMissing experiments:")
            for params in missing_experiments:
                print(f"  - {params.workload.value}/{params.pinning.value}/{params.scheduler.value}")

        print("=" * 60)
        print("RUNNING EXPERIMENTS")
        print("=" * 60)

        # Step 5: Run missing experiments
        for i, params in enumerate(missing_experiments):
            print(f"\nProgress: {i+1}/{len(missing_experiments)} (missing experiments)")

            result = self._run_experiment(params.workload, params.pinning, params.scheduler)
            self.results.append(result)

            # Save intermediate results after each experiment
            self._save_results()

        if missing_experiments:
            print(f"\n{'='*60}")
            print(f"Completed {len(missing_experiments)} new experiments!")
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

    def _load_result_from_run(self, run_dir: Path) -> Optional[ExperimentResult]:
        """Load experiment result from a completed run directory."""
        # Load params
        params = self._load_params_from_run(run_dir)
        if not params:
            return None

        # Parse stress-ng YAML files
        cpu_metrics: Optional[StressMetrics] = None
        mem_metrics: Optional[StressMetrics] = None

        if params.workload in [WorkloadType.BOTH, WorkloadType.CPU]:
            cpu_yaml = run_dir / "metrics_cpu.yaml"
            cpu_metrics = self._parse_stress_yaml(str(cpu_yaml))

        if params.workload in [WorkloadType.BOTH, WorkloadType.MEM]:
            mem_yaml = run_dir / "metrics_mem.yaml"
            mem_metrics = self._parse_stress_yaml(str(mem_yaml))

        # Parse perf JSON output
        perf_output_file = run_dir / "perf.json"
        perf_metrics = self._parse_perf_json(perf_output_file)

        # Create result
        return ExperimentResult(
            params=params,
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

    def _load_all_results(self) -> List[ExperimentResult]:
        """Load all experiment results from completed run directories."""
        all_results: List[ExperimentResult] = []

        if not self.results_dir.exists():
            return all_results

        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                result = self._load_result_from_run(run_dir)
                if result:
                    all_results.append(result)

        return all_results

    def _save_results(self) -> None:
        """Save all results to CSV file."""
        # Load all results from completed runs
        all_results = self._load_all_results()

        if not all_results:
            print("No results to save!")
            return

        # Convert dataclasses to dictionaries with string enum values for CSV
        results_dicts: List[Dict[str, Any]] = []
        for result in all_results:
            results_dicts.append(result.to_dict())

        df = pd.DataFrame(results_dicts)
        csv_file = self.results_dir / "experiment_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file} ({len(all_results)} experiments)")

    def analyze_and_plot(self) -> None:
        """Analyze results and create visualization."""
        # Load all results from completed runs (not just those run in this session)
        all_results = self._load_all_results()

        if not all_results:
            print("No completed results found for analysis!")
            return

        print(f"Analyzing {len(all_results)} completed experiments...")

        # Convert dataclasses to dictionaries for analysis
        results_dicts: List[Dict[str, Any]] = []
        for result in all_results:
            results_dicts.append(result.to_dict())

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
        # Sort by workload then pinning for consistent display
        df_sorted = df.sort_values(['workload', 'pinning'])
        print(df_sorted[summary_cols].round(1).to_string(index=False))

    def _create_plots(self, df: pd.DataFrame, max_cpu_persec: float, max_mem_persec: float) -> None:
        """Create improved stacked bar chart visualization."""
        # Get unique schedulers from the data
        schedulers = sorted(df['scheduler'].unique())
        num_schedulers = len(schedulers)
        
        fig, axes = plt.subplots(1, num_schedulers, figsize=(12 * num_schedulers, 8))
        if num_schedulers == 1:
            axes = [axes]  # Make it a list for consistent indexing
            
        fig.suptitle(f'CPU Scheduling Experiment Results\n'
                     f'100% CPU = {max_cpu_persec:,.0f} ops/sec, '
                     f'100% MEM = {max_mem_persec:,.0f} ops/sec')

        # Consistent colors for CPU and MEM across all bars
        cpu_color = '#2E86AB'  # Blue for CPU
        mem_color = '#A23B72'  # Purple for MEM

        # Plot: Stacked bar chart with consistent colors
        workloads = ['both', 'cpu', 'mem']
        pinning_strategies = ['none', 'spread', 'half']

        for sched_idx, scheduler in enumerate(schedulers):
            ax = axes[sched_idx]
            sched_df = df[df['scheduler'] == scheduler]
            
            x_pos = np.arange(len(workloads))
            width = 0.25

            for i, pinning in enumerate(pinning_strategies):
                subset = sched_df[sched_df['pinning'] == pinning]

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
                                label='CPU' if i == 0 and sched_idx == 0 else '',
                                color=cpu_color, alpha=0.8,
                                edgecolor='black', linewidth=0.5)
                mem_bars = ax.bar(x_offset, mem_values, width, bottom=cpu_values,
                                label='MEM' if i == 0 and sched_idx == 0 else '',
                                color=mem_color, alpha=0.8,
                                edgecolor='black', linewidth=0.5)

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

                # Add combined throughput labels on top of all bars
                for j, (cpu_bar, cpu_val, mem_val) in enumerate(zip(cpu_bars, cpu_values, mem_values)):
                    combined_val = cpu_val + mem_val
                    # Place label above the bar with some padding
                    ax.text(cpu_bar.get_x() + cpu_bar.get_width()/2, 
                           combined_val + 5,  # 5 units above the bar
                           f'{combined_val:.0f}%', ha='center', va='bottom',
                           fontweight='bold', fontsize=10, color='black')

            # Update x-axis labels to show pinning strategies
            workload_labels = [f'{wl}\n(none/spread/half)' for wl in workloads]

            ax.set_xlabel('Workload Type (Pinning Strategies Left to Right)')
            ax.set_ylabel('Normalized Performance (%)')
            ax.set_title(f'{scheduler.upper()} Scheduler')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(workload_labels)
            if sched_idx == 0:  # Only show legend on first subplot
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
