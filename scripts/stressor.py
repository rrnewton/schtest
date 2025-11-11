#!/usr/bin/env python3
"""
Stressor Module
Abstract interface for stress testing tools with concrete implementations.
"""

import os
import subprocess
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StressMetrics:
    """Typed structure for stress test metrics"""
    bogo_ops: int
    bogo_ops_per_sec_cpu_time: float
    real_time: float


class Stressor(ABC):
    """Abstract base class for stress testing tools."""

    def __init__(self, duration: int, output_dir: Path):
        """Initialize stressor with duration and output directory.

        Args:
            duration: Test duration in seconds
            output_dir: Directory where output files will be written
        """
        self.duration = duration
        self.output_dir = output_dir
        self.cpu_stressors: List[Dict[str, Any]] = []
        self.mem_stressors: List[Dict[str, Any]] = []

    def add_cpu_stressor(self, count: int, cpu_list: Optional[List[int]] = None, **kwargs: Any) -> 'Stressor':
        """Add a CPU stress test.

        Args:
            count: Number of CPU stressor instances
            cpu_list: List of CPU numbers to pin to, or None for no pinning
            **kwargs: Additional stressor-specific parameters

        Returns:
            Self for method chaining
        """
        self.cpu_stressors.append({
            'count': count,
            'cpu_list': cpu_list,
            'kwargs': kwargs
        })
        return self

    def add_mem_stressor(
        self,
        count: int,
        cpu_list: Optional[List[int]] = None,
        bytes_per_worker: str = '1g',
        **kwargs: Any
    ) -> 'Stressor':
        """Add a memory stress test.

        Args:
            count: Number of memory stressor instances
            cpu_list: List of CPU numbers to pin to, or None for no pinning
            bytes_per_worker: Memory size per worker (e.g., '1g', '512m')
            **kwargs: Additional stressor-specific parameters

        Returns:
            Self for method chaining
        """
        self.mem_stressors.append({
            'count': count,
            'cpu_list': cpu_list,
            'bytes_per_worker': bytes_per_worker,
            'kwargs': kwargs
        })
        return self

    @abstractmethod
    def execute(self) -> Tuple[Optional[StressMetrics], Optional[StressMetrics]]:
        """Execute the stress test and return (cpu_metrics, mem_metrics).

        This method should:
        1. Generate the stress test script
        2. Execute the script
        3. Parse the results
        4. Return the metrics

        Returns:
            Tuple of (cpu_metrics, mem_metrics) where each can be None if not applicable
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Tuple[Optional[StressMetrics], Optional[StressMetrics]]:
        """Parse and return metrics from already-executed stress test.

        This is used for loading results from disk without re-executing.

        Returns:
            Tuple of (cpu_metrics, mem_metrics) where each can be None if not applicable
        """
        pass


class StressNGStressor(Stressor):
    """Stress-ng based stressor implementation."""

    def __init__(self, duration: int, output_dir: Path):
        """Initialize stress-ng stressor.

        Args:
            duration: Test duration in seconds
            output_dir: Directory where YAML output files will be written
        """
        super().__init__(duration, output_dir)

    def _format_cpu_list(self, cpu_list: Optional[List[int]]) -> str:
        """Format CPU list for taskset parameter."""
        if not cpu_list:
            return ""
        return f"--taskset {','.join(map(str, cpu_list))}"

    def _get_cpu_params(self, count: int, **kwargs: Any) -> str:
        """Get stress-ng CPU workload parameters."""
        method = kwargs.get('method', 'int64')
        return f"--cpu {count} --cpu-method {method}"

    def _get_mem_params(self, count: int, bytes_per_worker: str, **kwargs: Any) -> str:
        """Get stress-ng memory workload parameters."""
        method = kwargs.get('method', 'ror')
        keep = '--vm-keep' if kwargs.get('keep', True) else ''
        return f"--vm {count} {keep} --vm-method {method} --vm-bytes {bytes_per_worker}"

    def _create_script(self) -> str:
        """Generate bash script for stress-ng configuration."""
        script_lines = ["#!/bin/bash", "set -xeuo pipefail"]

        # Determine if we need to run stressors in parallel
        has_multiple_stressor_types = bool(self.cpu_stressors and self.mem_stressors)

        # Add CPU stressors
        for i, cpu_stress in enumerate(self.cpu_stressors):
            output_file = self.output_dir / f"metrics_cpu_{i}.yaml"
            taskset = self._format_cpu_list(cpu_stress['cpu_list'])
            params = self._get_cpu_params(cpu_stress['count'], **cpu_stress['kwargs'])

            cmd = f"stress-ng --metrics -t {self.duration} --yaml {output_file} {params} {taskset}"

            # If we have both CPU and memory stressors, run CPU stressors in background
            if has_multiple_stressor_types:
                script_lines.append(f"({cmd}) &")
            else:
                script_lines.append(f"{cmd};")

        # Add memory stressors
        for i, mem_stress in enumerate(self.mem_stressors):
            output_file = self.output_dir / f"metrics_mem_{i}.yaml"
            taskset = self._format_cpu_list(mem_stress['cpu_list'])
            params = self._get_mem_params(
                mem_stress['count'],
                mem_stress['bytes_per_worker'],
                **mem_stress['kwargs']
            )

            cmd = f"stress-ng --metrics -t {self.duration} --yaml {output_file} {params} {taskset}"
            script_lines.append(f"{cmd};")

        return "\n".join(script_lines) + "\n"

    def _parse_yaml(self, yaml_file: Path) -> Optional[StressMetrics]:
        """Parse a stress-ng YAML output file."""
        if not yaml_file.exists():
            return None
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            metrics_list = data.get('metrics', [{}])
            if not metrics_list or not isinstance(metrics_list, list):
                return None
            metrics = metrics_list[0]
            return StressMetrics(
                bogo_ops=metrics.get('bogo-ops', 0),
                bogo_ops_per_sec_cpu_time=metrics.get('bogo-ops-per-second-usr-sys-time', 0.0),
                real_time=metrics.get('wall-clock-time', 0.0)
            )
        except Exception as e:
            print(f"Error parsing YAML file {yaml_file}: {e}")
            return None

    def execute(self) -> Tuple[Optional[StressMetrics], Optional[StressMetrics]]:
        """Execute the stress test and return metrics."""
        # Generate and write script
        script_content = self._create_script()
        script_file = self.output_dir / "stress.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        os.chmod(script_file, 0o755)

        # Execute script
        result = subprocess.run(
            [str(script_file)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Stress test failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")

        # Parse and return metrics
        return self.get_metrics()

    def get_metrics(self) -> Tuple[Optional[StressMetrics], Optional[StressMetrics]]:
        """Parse and return metrics from already-executed stress test."""
        cpu_metrics = None
        mem_metrics = None

        # Check for CPU metrics based on whether CPU stressors were added
        if self.cpu_stressors:
            cpu_yaml = self.output_dir / "metrics_cpu_0.yaml"
            cpu_metrics = self._parse_yaml(cpu_yaml)

        # Check for memory metrics based on whether memory stressors were added
        if self.mem_stressors:
            mem_yaml = self.output_dir / "metrics_mem_0.yaml"
            mem_metrics = self._parse_yaml(mem_yaml)

        return cpu_metrics, mem_metrics


def get_l3_fairshare(cpu: int = 0, percentage: float = 0.95) -> int:
    """Calculate L3 cache fairshare per core.

    Args:
        cpu: CPU number to check for L3 cache info (default: 0)
        percentage: Percentage of fairshare to return (default: 0.95 = 95%)

    Returns:
        Number of bytes representing the percentage of L3 fairshare per core

    Raises:
        FileNotFoundError: If L3 cache information cannot be found
        ValueError: If L3 cache information is invalid
    """
    # Find L3 cache index in /sys
    cache_base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    if not cache_base.exists():
        raise FileNotFoundError(f"Cache information not found for CPU {cpu}")

    l3_index = None
    for index_dir in cache_base.iterdir():
        if index_dir.is_dir() and index_dir.name.startswith("index"):
            level_file = index_dir / "level"
            if level_file.exists():
                with open(level_file) as f:
                    if f.read().strip() == "3":
                        l3_index = index_dir
                        break

    if not l3_index:
        raise FileNotFoundError(f"L3 cache not found for CPU {cpu}")

    # Read L3 cache size
    size_file = l3_index / "size"
    with open(size_file) as f:
        size_str = f.read().strip()
        # Parse size like "32768K" or "128M"
        if size_str.endswith('K'):
            l3_size = int(size_str[:-1]) * 1024
        elif size_str.endswith('M'):
            l3_size = int(size_str[:-1]) * 1024 * 1024
        elif size_str.endswith('G'):
            l3_size = int(size_str[:-1]) * 1024 * 1024 * 1024
        else:
            l3_size = int(size_str)

    # Read shared CPU list to count cores sharing this L3
    shared_cpus_file = l3_index / "shared_cpu_list"
    with open(shared_cpus_file) as f:
        shared_cpus_str = f.read().strip()
        # Parse ranges like "0-7,32-39"
        cpu_count = 0
        for part in shared_cpus_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                cpu_count += (end - start + 1)
            else:
                cpu_count += 1

    if cpu_count == 0:
        raise ValueError(f"Invalid shared CPU count for L3 cache: {shared_cpus_str}")

    # Calculate fairshare per core
    fairshare_per_core = l3_size / cpu_count
    return int(fairshare_per_core * percentage)


def find_rt_app() -> str:
    """Find rt-app binary, checking PATH and relative paths.

    Returns:
        Path to rt-app executable

    Raises:
        FileNotFoundError: If rt-app cannot be found
    """
    # First check if rt-app is on PATH
    rt_app_path = shutil.which('rt-app')
    if rt_app_path:
        return rt_app_path

    # Find git repo root to check for ../rt-app
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            check=True
        )
        repo_root = Path(result.stdout.strip())

        # Check for ../rt-app/src/rt-app relative to repo root
        relative_rt_app = repo_root.parent / 'rt-app' / 'src' / 'rt-app'
        if relative_rt_app.exists() and relative_rt_app.is_file():
            return str(relative_rt_app)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repo or git not available, continue to error
        pass

    # Could not find rt-app
    raise FileNotFoundError(
        "rt-app not found. Please either:\n"
        "  1. Install rt-app and ensure it's on PATH, or\n"
        "  2. Build rt-app in ../rt-app/ relative to the repo root"
    )


class RTAppStressor(Stressor):
    """rt-app based stressor implementation."""

    def __init__(self, duration: int, output_dir: Path, rt_app_path: Optional[str] = None):
        """Initialize rt-app stressor.

        Args:
            duration: Test duration in seconds
            output_dir: Directory where JSON config and output files will be written
            rt_app_path: Path to rt-app executable (auto-detected if None)

        Raises:
            FileNotFoundError: If rt-app cannot be found
        """
        super().__init__(duration, output_dir)
        self.rt_app_path = rt_app_path or find_rt_app()

    def add_mem_stressor(
        self,
        count: int,
        cpu_list: Optional[List[int]] = None,
        bytes_per_worker: Optional[str] = None,
        **kwargs: Any
    ) -> 'RTAppStressor':
        """Add a memory stress test with L3-aware default sizing.

        Args:
            count: Number of memory stressor instances
            cpu_list: List of CPU numbers to pin to, or None for no pinning
            bytes_per_worker: Memory size per worker (e.g., '1g', '512m').
                            If None, defaults to 95% of L3 fairshare per core.
            **kwargs: Additional stressor-specific parameters

        Returns:
            Self for method chaining
        """
        # Default to 95% of L3 fairshare if not specified
        if bytes_per_worker is None:
            # Determine which CPU to check based on cpu_list
            check_cpu = cpu_list[0] if cpu_list else 0
            try:
                fairshare_bytes = get_l3_fairshare(cpu=check_cpu, percentage=0.95)
                # Convert to string format (e.g., "1992294" -> "1992294")
                bytes_per_worker = str(fairshare_bytes)
            except (FileNotFoundError, ValueError) as e:
                # Fall back to 1g if L3 detection fails
                print(f"Warning: Could not detect L3 cache, using default 1g: {e}")
                bytes_per_worker = '1g'

        # Call parent implementation
        super().add_mem_stressor(count, cpu_list, bytes_per_worker, **kwargs)
        return self

    def _parse_memory_size(self, size_str: str) -> int:
        """Parse memory size string (e.g., '1g', '512m') to bytes.

        Args:
            size_str: Size string like '1g', '512m', '1024k'

        Returns:
            Size in bytes
        """
        size_str = size_str.lower().strip()
        multipliers = {
            'k': 1024,
            'm': 1024 * 1024,
            'g': 1024 * 1024 * 1024,
        }

        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(size_str[:-1]) * multiplier

        # No suffix, assume bytes
        return int(size_str)

    def _create_json_config(self) -> Dict[str, Any]:
        """Generate rt-app JSON configuration."""
        tasks = {}

        # Determine max memory buffer size from all memory stressors
        max_mem_buffer = 4 * 1024 * 1024  # Default 4MB
        if self.mem_stressors:
            max_mem_buffer = max(
                self._parse_memory_size(mem['bytes_per_worker'])
                for mem in self.mem_stressors
            )

        # Add CPU stressor threads
        for i, cpu_stress in enumerate(self.cpu_stressors):
            for instance in range(cpu_stress['count']):
                thread_name = f"cpu_{i}_{instance}"
                thread_config = {
                    "loop": -1,  # Run until duration expires
                    "run": 100000,  # Run for 100ms (100% CPU utilization)
                }

                # Add CPU affinity if specified
                if cpu_stress['cpu_list']:
                    thread_config["cpus"] = cpu_stress['cpu_list']

                tasks[thread_name] = thread_config

        # Add memory stressor threads
        for i, mem_stress in enumerate(self.mem_stressors):
            mem_buffer_size = self._parse_memory_size(mem_stress['bytes_per_worker'])

            for instance in range(mem_stress['count']):
                thread_name = f"mem_{i}_{instance}"
                thread_config = {
                    "loop": -1,  # Run until duration expires
                    "mem": 10000,  # Memory operations for 10ms
                    "sleep": 10000,  # Sleep for 10ms (50% duty cycle)
                }

                # Add CPU affinity if specified
                if mem_stress['cpu_list']:
                    thread_config["cpus"] = mem_stress['cpu_list']

                tasks[thread_name] = thread_config

        # Build complete configuration
        config = {
            "tasks": tasks,
            "global": {
                "duration": self.duration,
                "calibration": "CPU0",
                "default_policy": "SCHED_OTHER",
                "pi_enabled": False,
                "lock_pages": False,
                "mem_buffer_size": max_mem_buffer,
            }
        }

        return config

    def execute(self) -> Tuple[Optional[StressMetrics], Optional[StressMetrics]]:
        """Execute the stress test and return metrics."""
        # Generate and write JSON config
        config = self._create_json_config()
        config_file = self.output_dir / "rt-app-config.json"

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Execute rt-app
        result = subprocess.run(
            [self.rt_app_path, str(config_file)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: rt-app failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            print(f"stdout: {result.stdout}")

        # rt-app doesn't provide the same metrics as stress-ng
        # Return None for now
        return self.get_metrics()

    def get_metrics(self) -> Tuple[Optional[StressMetrics], Optional[StressMetrics]]:
        """Parse and return metrics from already-executed stress test.

        Note: rt-app doesn't provide bogo-ops metrics like stress-ng.
        This method returns None for both CPU and memory metrics.
        If we need metrics later, we can parse rt-app's log files.
        """
        # rt-app generates log files but they don't contain bogo-ops
        # Return None to indicate no metrics available
        cpu_metrics = None if self.cpu_stressors else None
        mem_metrics = None if self.mem_stressors else None

        return cpu_metrics, mem_metrics
