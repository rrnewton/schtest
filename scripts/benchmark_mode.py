#!/usr/bin/env python3
"""
Benchmark Mode Script

This script locks down the system for consistent benchmarking by:
1. Locking CPU frequency scaling at a low frequency to prevent thermal throttling
2. (TODO) Sequestering other processes to CPU0 for isolated benchmarking on CPU1..N

Usage:
    sudo ./benchmark_mode.py

The script will:
- Acquire a lockfile in /tmp to prevent multiple instances
- Configure the system for benchmarking
- Print "READY_FOR_BENCHMARKING" when setup is complete
- Run until killed (SIGINT/SIGTERM)
- Clean up all changes on exit
"""

import os
import sys
import signal
import time
import fcntl
import atexit
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


# Constants
LOCKFILE_PATH = "/tmp/benchmark_mode.lock"
CPUFREQ_BASE = Path("/sys/devices/system/cpu")


@dataclass
class CPUState:
    """Stores original CPU state for restoration."""
    cpu_id: int
    governor: str
    min_freq: str
    max_freq: str


class BenchmarkMode:
    """Manages system configuration for benchmarking."""

    def __init__(self):
        self.lockfile: Optional[int] = None
        self.original_states: List[CPUState] = []
        self.cleanup_done = False

    def _acquire_lockfile(self) -> None:
        """Acquire exclusive lockfile to prevent multiple instances."""
        try:
            # Open lockfile (create if doesn't exist)
            self.lockfile = os.open(LOCKFILE_PATH, os.O_CREAT | os.O_RDWR, 0o644)

            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID to the lockfile
            os.ftruncate(self.lockfile, 0)
            os.write(self.lockfile, f"{os.getpid()}\n".encode())

            print(f"Acquired lockfile: {LOCKFILE_PATH}")

        except BlockingIOError:
            print(f"ERROR: Another instance of benchmark_mode is already running.", file=sys.stderr)
            print(f"Lockfile: {LOCKFILE_PATH}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to acquire lockfile: {e}", file=sys.stderr)
            sys.exit(1)

    def _release_lockfile(self) -> None:
        """Release lockfile."""
        if self.lockfile is not None:
            try:
                fcntl.flock(self.lockfile, fcntl.LOCK_UN)
                os.close(self.lockfile)
                os.unlink(LOCKFILE_PATH)
                print("Released lockfile")
            except Exception as e:
                print(f"Warning: Error releasing lockfile: {e}", file=sys.stderr)
            finally:
                self.lockfile = None

    def _get_cpu_numbers(self) -> List[int]:
        """Get list of all CPU numbers on the system."""
        cpus = []
        for cpu_dir in sorted(CPUFREQ_BASE.glob("cpu[0-9]*")):
            cpu_num = int(cpu_dir.name[3:])
            cpufreq_dir = cpu_dir / "cpufreq"
            if cpufreq_dir.exists():
                cpus.append(cpu_num)
        return cpus

    def _read_cpu_file(self, cpu_id: int, filename: str) -> str:
        """Read a cpufreq file for a specific CPU."""
        path = CPUFREQ_BASE / f"cpu{cpu_id}" / "cpufreq" / filename
        try:
            return path.read_text().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read {path}: {e}")

    def _write_cpu_file(self, cpu_id: int, filename: str, value: str) -> None:
        """Write to a cpufreq file for a specific CPU."""
        path = CPUFREQ_BASE / f"cpu{cpu_id}" / "cpufreq" / filename
        try:
            path.write_text(value)
        except Exception as e:
            raise RuntimeError(f"Failed to write '{value}' to {path}: {e}")

    def _save_cpu_state(self, cpu_id: int) -> CPUState:
        """Save current state of a CPU."""
        return CPUState(
            cpu_id=cpu_id,
            governor=self._read_cpu_file(cpu_id, "scaling_governor"),
            min_freq=self._read_cpu_file(cpu_id, "scaling_min_freq"),
            max_freq=self._read_cpu_file(cpu_id, "scaling_max_freq"),
        )

    def _restore_cpu_state(self, state: CPUState) -> None:
        """Restore a CPU to its original state."""
        try:
            # Restore governor first
            self._write_cpu_file(state.cpu_id, "scaling_governor", state.governor)
            # Then restore frequency limits
            self._write_cpu_file(state.cpu_id, "scaling_min_freq", state.min_freq)
            self._write_cpu_file(state.cpu_id, "scaling_max_freq", state.max_freq)
            print(f"Restored CPU{state.cpu_id}: governor={state.governor}, "
                  f"min={state.min_freq}, max={state.max_freq}")
        except Exception as e:
            print(f"Warning: Failed to restore CPU{state.cpu_id}: {e}", file=sys.stderr)

    def _get_lowest_frequency(self, cpu_id: int) -> str:
        """Get the lowest available frequency for a CPU."""
        try:
            # Try to read available frequencies
            freqs_str = self._read_cpu_file(cpu_id, "scaling_available_frequencies")
            frequencies = [int(f) for f in freqs_str.split()]
            if frequencies:
                return str(min(frequencies))
        except Exception:
            pass

        # Fallback: use cpuinfo_min_freq
        try:
            return self._read_cpu_file(cpu_id, "cpuinfo_min_freq")
        except Exception as e:
            raise RuntimeError(f"Could not determine lowest frequency for CPU{cpu_id}: {e}")

    def _lock_cpu_frequency(self) -> None:
        """Lock all CPUs to lowest frequency to prevent thermal throttling."""
        print("\nLocking CPU frequencies...")

        cpus = self._get_cpu_numbers()
        if not cpus:
            raise RuntimeError("No CPUs with cpufreq support found!")

        print(f"Found {len(cpus)} CPUs with frequency scaling support")

        for cpu_id in cpus:
            # Save original state
            original = self._save_cpu_state(cpu_id)
            self.original_states.append(original)

            # Get lowest frequency
            lowest_freq = self._get_lowest_frequency(cpu_id)

            print(f"CPU{cpu_id}: current governor={original.governor}, "
                  f"freq range=[{original.min_freq}, {original.max_freq}]")
            print(f"CPU{cpu_id}: locking to {lowest_freq} kHz")

            # Set governor to userspace (allows manual frequency setting)
            # If userspace is not available, use powersave as fallback
            try:
                self._write_cpu_file(cpu_id, "scaling_governor", "userspace")
                governor = "userspace"
            except Exception:
                print(f"  Warning: userspace governor not available, trying powersave")
                try:
                    self._write_cpu_file(cpu_id, "scaling_governor", "powersave")
                    governor = "powersave"
                except Exception as e:
                    raise RuntimeError(f"Failed to set governor for CPU{cpu_id}: {e}")

            # Lock frequency to lowest value
            # For userspace governor, we set scaling_setspeed
            # For powersave governor, we set both min and max to the same value
            if governor == "userspace":
                try:
                    self._write_cpu_file(cpu_id, "scaling_setspeed", lowest_freq)
                except Exception as e:
                    print(f"  Warning: Could not set scaling_setspeed: {e}")

            # Always set min and max freq to lock the frequency
            self._write_cpu_file(cpu_id, "scaling_min_freq", lowest_freq)
            self._write_cpu_file(cpu_id, "scaling_max_freq", lowest_freq)

            # Verify
            current_min = self._read_cpu_file(cpu_id, "scaling_min_freq")
            current_max = self._read_cpu_file(cpu_id, "scaling_max_freq")
            current_gov = self._read_cpu_file(cpu_id, "scaling_governor")

            print(f"  Verified: governor={current_gov}, freq=[{current_min}, {current_max}]")

    def _restore_all_cpus(self) -> None:
        """Restore all CPUs to their original state."""
        if not self.original_states:
            return

        print("\nRestoring CPU frequencies...")
        for state in self.original_states:
            self._restore_cpu_state(state)

        self.original_states.clear()

    def cleanup(self) -> None:
        """Clean up all changes made to the system."""
        if self.cleanup_done:
            return

        print("\n" + "="*60)
        print("CLEANUP: Restoring system to original state")
        print("="*60)

        self._restore_all_cpus()
        self._release_lockfile()

        self.cleanup_done = True
        print("Cleanup complete")

    def setup(self) -> None:
        """Set up the system for benchmarking."""
        print("="*60)
        print("BENCHMARK MODE SETUP")
        print("="*60)

        # Check if running as root
        if os.geteuid() != 0:
            print("ERROR: This script must be run as root (use sudo)", file=sys.stderr)
            sys.exit(1)

        # Acquire lockfile
        self._acquire_lockfile()

        # Lock CPU frequencies
        self._lock_cpu_frequency()

        # TODO: Sequester processes to CPU0
        # This will be implemented in a future version

        print("\n" + "="*60)
        print("READY_FOR_BENCHMARKING")
        print("="*60)
        print("\nBenchmark mode is active. Press Ctrl+C to exit and restore system.")
        sys.stdout.flush()

    def run(self) -> None:
        """Run until killed."""
        try:
            # Just sleep forever until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal")


def signal_handler(signum: int, frame) -> None:
    """Handle termination signals."""
    print(f"\nReceived signal {signum}")
    sys.exit(0)


def main() -> None:
    """Main entry point."""
    benchmark = BenchmarkMode()

    # Register cleanup handler
    atexit.register(benchmark.cleanup)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Set up benchmark mode
        benchmark.setup()

        # Run until killed
        benchmark.run()

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
