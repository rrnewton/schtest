#!/usr/bin/env python3
"""
CPU Scheduling Experiment Script
Runs stress-ng workloads with different thread pinning strategies
and analyzes performance metrics.
"""

import atexit
import functools
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

# Import topology parser
from parse_topo import parse_topology, Machine
# Import scheduler monitor
from sched_monitor import SchedMonitor, SchedulerType
# Import stressor classes
from stressor import Stressor, StressNGStressor, StressMetrics


# Ownership preservation for running as root
def _get_target_ownership(path: Path) -> Tuple[int, int]:
    """Get the target uid/gid for files in the given path.

    If running as root, we want to preserve the ownership of the parent directory
    to avoid polluting user-owned directories with root-owned files.

    Args:
        path: Path to check (or its parent if it doesn't exist)

    Returns:
        Tuple of (uid, gid) for the target ownership
    """
    if os.geteuid() != 0:
        # Not running as root, no need to change ownership
        return (-1, -1)

    # Find the first existing parent directory
    check_path = path
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent

    if check_path.exists():
        stat_info = check_path.stat()
        return (stat_info.st_uid, stat_info.st_gid)
    else:
        # Fallback: use the user who invoked sudo, if available
        sudo_uid = os.environ.get('SUDO_UID')
        sudo_gid = os.environ.get('SUDO_GID')
        if sudo_uid and sudo_gid:
            return (int(sudo_uid), int(sudo_gid))

    return (-1, -1)


def _fix_ownership_recursive(path: Path, uid: int, gid: int) -> None:
    """Recursively fix ownership of a directory and all its contents.

    Args:
        path: Path to fix
        uid: Target user ID
        gid: Target group ID
    """
    if uid < 0 or gid < 0:
        # No ownership change needed
        return

    if not path.exists():
        return

    try:
        print(f"\nRestoring ownership of {path} to uid={uid}, gid={gid}...")

        # Fix the path itself
        os.chown(path, uid, gid)

        # If it's a directory, recursively fix contents
        if path.is_dir():
            for item in path.rglob('*'):
                try:
                    os.chown(item, uid, gid)
                except (OSError, PermissionError) as e:
                    print(f"Warning: Could not fix ownership of {item}: {e}")

        print("Ownership restored successfully")

    except (OSError, PermissionError) as e:
        print(f"Warning: Could not fix ownership of {path}: {e}")


# Configuration
EXPERIMENT_DURATION = 6  # seconds
RESULTS_DIR = Path("./results")
STRESS_SCRIPT = "./stress.sh"
SCX_DIR = Path("../../scx")

@functools.total_ordering
class WorkloadType(Enum):
    BOTH = "both"
    CPU = "cpu"
    MEM = "mem"

    # Custom ordering to do the baseline one-workload tests first.
    def __lt__(self, other: object) -> bool:
        if isinstance(other, WorkloadType):
            order = [WorkloadType.CPU, WorkloadType.MEM, WorkloadType.BOTH]
            return order.index(self) < order.index(other)
        return NotImplemented

class PinningStrategy(Enum):
    """Enum for thread pinning strategies"""
    NONE = "none"
    SPREAD = "spread"
    HALF = "half"

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
        self.sched_monitor: Optional[SchedMonitor] = None
        self.current_scheduler: SchedulerType = SchedulerType.DEFAULT  # Track currently active scheduler

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

    def _ensure_scheduler(self, scheduler: SchedulerType) -> None:
        """Ensure the requested scheduler is active, switching if necessary.

        This is a lazy operation - only switches if the current scheduler
        doesn't match the requested one.
        """
        # If already running the right scheduler, do nothing
        if self.current_scheduler == scheduler:
            print(f"Using already-started scheduler: {scheduler.value}")
            return

        # Stop current scheduler if it's not DEFAULT
        if self.current_scheduler != SchedulerType.DEFAULT:
            self._stop_current_scheduler()

        # Start new scheduler if it's not DEFAULT
        if scheduler != SchedulerType.DEFAULT:
            if scheduler == SchedulerType.SCX_LAVD:
                scheduler_path = SCX_DIR / "target/release/scx_lavd"
                self.sched_monitor = SchedMonitor(scheduler, scheduler_path)
                self.sched_monitor.start()
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")

        # Update current scheduler
        self.current_scheduler = scheduler
        print(f"Active scheduler: {scheduler.value}")

    def _get_cpu_list_for_pinning(self, pinning: PinningStrategy) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Get CPU lists for CPU and memory workloads based on pinning strategy.

        Returns:
            Tuple of (cpu_list, mem_list) where each can be None for no pinning
        """
        if pinning == PinningStrategy.NONE:
            return None, None
        elif pinning == PinningStrategy.SPREAD:
            # Use split_hyperthreads to spread across all cores
            try:
                cpu_list, mem_list = self.topology.split_hyperthreads()
                print(f"SPREAD strategy: CPU cores {cpu_list}, MEM cores {mem_list}")
                return cpu_list, mem_list
            except ValueError as e:
                print(f"Warning: Could not split hyperthreads ({e}), falling back to no pinning")
                return None, None
        elif pinning == PinningStrategy.HALF:
            # Use split_physical to occupy only half of the cores physically
            try:
                _, cpu_list, mem_list = self.topology.split_physical()
                print(f"HALF strategy: CPU cores {cpu_list}, MEM cores {mem_list}")
                return cpu_list, mem_list
            except ValueError as e:
                print(f"Warning: Could not split physical cores ({e}), falling back to no pinning")
                return None, None
        else:
            raise ValueError(f"Unknown pinning strategy: {pinning}")

    def _create_stressor(self, workload: WorkloadType, pinning: PinningStrategy, run_dir: Path) -> StressNGStressor:
        """Create and configure stressor for given configuration."""
        P = self.num_cores

        # Get CPU lists based on pinning strategy
        cpu_list, mem_list = self._get_cpu_list_for_pinning(pinning)

        # Create stressor with output directory
        stressor = StressNGStressor(EXPERIMENT_DURATION, run_dir)

        # Add stressors based on workload type
        if workload in [WorkloadType.BOTH, WorkloadType.CPU]:
            stressor.add_cpu_stressor(P, cpu_list, method='int64')

        if workload in [WorkloadType.BOTH, WorkloadType.MEM]:
            stressor.add_mem_stressor(P, mem_list, f'{P}g', method='ror', keep=True)

        return stressor

    def _stop_current_scheduler(self) -> None:
        """Stop the currently active scheduler."""
        if self.sched_monitor:
            self.sched_monitor.stop()
            self.sched_monitor = None
        self.current_scheduler = SchedulerType.DEFAULT

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

        # Create and configure stressor
        stressor = self._create_stressor(workload, pinning, run_dir)

        # Ensure correct scheduler is active (lazy switching)
        self._ensure_scheduler(scheduler)

        # Run experiment with perf wrapping the stressor execution
        perf_output_file = run_dir / "perf.json"
        stress_script = run_dir / "stress.sh"
        perf_cmd = [
            "perf", "stat", "-j",
            "-e", "instructions,cycles,cache-references,cache-misses",
            str(stress_script)
        ]

        print(f"Running: {' '.join(perf_cmd)}")

        # Execute stressor (generates script) but wrap with perf for the actual execution
        script_content = stressor._create_script()
        with open(stress_script, 'w') as f:
            f.write(script_content)
        os.chmod(stress_script, 0o755)

        # Run with perf
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

        # Parse results using stressor abstraction
        cpu_metrics, mem_metrics = stressor.get_metrics()

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

    # Stress-ng parsing is now handled by StressNGStressor

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
                else:
                    run_counter = 0

                params_file = run_dir / "params.yaml"

                if params_file.exists():
                    # Try to load params to validate completeness
                    params = self._load_params_from_run(run_dir)
                    if params:
                        complete_count += 1
                        max_run_counter = max(max_run_counter, run_counter)
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
        try:
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
        finally:
            # Always stop any running scheduler at the end
            if self.current_scheduler != SchedulerType.DEFAULT:
                print(f"\nCleaning up: Stopping {self.current_scheduler.value} scheduler...")
                self._stop_current_scheduler()

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

        # Create stressor configured for this workload type (for metrics parsing)
        stressor = self._create_stressor(params.workload, params.pinning, run_dir)
        cpu_metrics, mem_metrics = stressor.get_metrics()

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

        # Analyze timing skew for both workloads
        both_workloads = df[df['workload'] == 'both'].copy()
        if len(both_workloads) > 0 and 'real_time_cpu' in both_workloads.columns and 'real_time_mem' in both_workloads.columns:
            print("\nTiming Skew Analysis (for 'both' workload experiments):")
            print("=" * 60)

            # Calculate skew as the ratio of differences
            both_workloads['time_ratio'] = both_workloads['real_time_mem'] / both_workloads['real_time_cpu']
            both_workloads['skew_percent'] = (both_workloads['time_ratio'] - 1.0) * 100

            max_skew = both_workloads['skew_percent'].abs().max()
            median_skew = both_workloads['skew_percent'].median()
            mean_skew = both_workloads['skew_percent'].mean()

            print(f"Max skew:     {max_skew:6.1f}% (memory workload time vs CPU workload time)")
            print(f"Median skew:  {median_skew:6.1f}%")
            print(f"Average skew: {mean_skew:6.1f}%")

            # Show the worst offenders
            worst_case = both_workloads.loc[both_workloads['skew_percent'].abs().idxmax()]
            print(f"\nWorst case: {worst_case['scheduler']}/{worst_case['pinning']}")
            print(f"  CPU time: {worst_case['real_time_cpu']:.2f}s, MEM time: {worst_case['real_time_mem']:.2f}s")
            print(f"  Skew: {worst_case['skew_percent']:.1f}%")

        # Print summary table
        print("\nSummary Results:")
        summary_cols = ['scheduler', 'workload', 'pinning', 'cpu_normalized', 'mem_normalized', 'combined_tput']
        # Sort by scheduler first, then workload and pinning for consistent display
        df_sorted = df.sort_values(['scheduler', 'workload', 'pinning'])
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
                cache_miss_rates: List[float] = []

                for workload in workloads:
                    row = subset[subset['workload'] == workload]
                    if len(row) > 0:
                        cpu_values.append(float(row['cpu_normalized'].iloc[0]))
                        mem_values.append(float(row['mem_normalized'].iloc[0]))

                        # Calculate cache miss rate
                        cache_refs = float(row['cache_refs'].iloc[0])
                        cache_misses = float(row['cache_misses'].iloc[0])
                        if cache_refs > 0:
                            miss_rate = (cache_misses / cache_refs) * 100
                        else:
                            miss_rate = 0.0
                        cache_miss_rates.append(miss_rate)
                    else:
                        cpu_values.append(0.0)
                        mem_values.append(0.0)
                        cache_miss_rates.append(0.0)

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
                for j, (cpu_bar, mem_bar, cpu_val, mem_val, miss_rate) in enumerate(zip(cpu_bars, mem_bars, cpu_values, mem_values, cache_miss_rates)):
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

                    # Cache miss rate label at bottom of bar
                    if miss_rate > 0:
                        ax.text(cpu_bar.get_x() + cpu_bar.get_width()/2, 3,
                               f'{miss_rate:.1f}% miss', ha='center', va='bottom',
                               fontsize=7, color='yellow', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'))

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

    # Set up ownership preservation if running as root
    if os.geteuid() == 0:
        target_uid, target_gid = _get_target_ownership(RESULTS_DIR)
        if target_uid >= 0 and target_gid >= 0:
            print(f"Running as root: will restore ownership to uid={target_uid}, gid={target_gid}")
            # Register exit handler to fix ownership on exit (success or error)
            atexit.register(lambda: _fix_ownership_recursive(RESULTS_DIR, target_uid, target_gid))

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
