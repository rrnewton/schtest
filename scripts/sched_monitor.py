#!/usr/bin/env python3
"""
Scheduler Monitor Library
Manages starting, stopping, and monitoring scheduler processes.
"""

import subprocess
import time
import select
from pathlib import Path
from typing import Optional, List
from enum import Enum


class SchedulerType(Enum):
    """Enum for scheduler types"""
    DEFAULT = "default"
    SCX_LAVD = "scx_lavd"


class SchedMonitor:
    """Monitor and manage scheduler processes."""

    def __init__(self, scheduler: SchedulerType, scheduler_path: Path) -> None:
        """Initialize scheduler monitor.

        Args:
            scheduler: Type of scheduler to monitor
            scheduler_path: Path to the scheduler binary
        """
        self.scheduler = scheduler
        self.scheduler_path = scheduler_path
        self.scheduler_proc: Optional[subprocess.Popen[str]] = None
        self.dmesg_proc: Optional[subprocess.Popen[str]] = None
        self.dmesg_lines: List[str] = []

        # Map scheduler types to their dmesg names
        self.scheduler_name_map = {
            SchedulerType.SCX_LAVD: "lavd"
        }

    def start(self, timeout: float = 30.0) -> subprocess.Popen[str]:
        """Start the scheduler and wait for it to be enabled.

        Args:
            timeout: Maximum time to wait for scheduler to start (seconds)

        Returns:
            The scheduler process

        Raises:
            RuntimeError: If scheduler fails to start
            FileNotFoundError: If scheduler binary doesn't exist
        """
        if self.scheduler == SchedulerType.DEFAULT:
            raise ValueError("Cannot start DEFAULT scheduler - it's always active")

        if not self.scheduler_path.exists():
            raise FileNotFoundError(f"Scheduler not found: {self.scheduler_path}")

        print(f"Starting scheduler: {self.scheduler_path}")

        # Get scheduler name for dmesg monitoring
        scheduler_name = self.scheduler_name_map.get(self.scheduler)
        if not scheduler_name:
            raise ValueError(f"Unknown scheduler type: {self.scheduler}")

        # Start dmesg monitoring
        self._start_dmesg_monitoring(scheduler_name)

        try:
            # Start the scheduler process with sudo
            self.scheduler_proc = subprocess.Popen(
                ["sudo", str(self.scheduler_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for scheduler to be enabled
            self._wait_for_sched_enabled(scheduler_name, timeout)

            return self.scheduler_proc

        except Exception:
            # Clean up on failure
            self._cleanup_dmesg()
            if self.scheduler_proc:
                self._stop_process(self.scheduler_proc)
            raise
        finally:
            # Always clean up dmesg monitoring
            self._cleanup_dmesg()

    def stop(self) -> None:
        """Stop the scheduler process.

        Sends SIGINT to allow the scheduler to cleanly unload itself from
        the kernel (same as Ctrl-C behavior).
        """
        if self.scheduler_proc is None:
            return

        print("Stopping scheduler...")

        # Stop the scheduler process (sends SIGINT first, then SIGTERM, then SIGKILL)
        self._stop_process(self.scheduler_proc)
        self.scheduler_proc = None
        print("Scheduler stopped")

    def _start_dmesg_monitoring(self, scheduler_name: str) -> None:
        """Start dmesg monitoring for the given scheduler."""
        self.dmesg_proc = subprocess.Popen(
            ["sudo", "dmesg", "-W"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        if self.dmesg_proc.stdout is None:
            self._cleanup_dmesg()
            raise RuntimeError("Failed to start dmesg monitoring")

        print(f"Started dmesg monitoring for {scheduler_name} scheduler...")

    def _wait_for_sched_enabled(self, scheduler_name: str, timeout: float) -> None:
        """Wait for scheduler enabled message in dmesg.

        Args:
            scheduler_name: Name of scheduler to look for in dmesg
            timeout: Maximum time to wait (seconds)

        Raises:
            RuntimeError: If timeout occurs or dmesg process dies
        """
        print(f"Waiting for {scheduler_name} scheduler to be enabled...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.dmesg_proc and self.dmesg_proc.poll() is not None:
                raise RuntimeError("dmesg monitoring process died")

            # Read with timeout to avoid blocking forever
            if self.dmesg_proc and self.dmesg_proc.stdout:
                ready, _, _ = select.select([self.dmesg_proc.stdout], [], [], 1.0)

                if ready:
                    line = self.dmesg_proc.stdout.readline()
                    if line:
                        stripped = line.strip()
                        print(f"dmesg: {stripped}")

                        # Keep last 10 lines for debugging
                        self.dmesg_lines.append(stripped)
                        if len(self.dmesg_lines) > 10:
                            self.dmesg_lines.pop(0)

                        # Look for scheduler enabled message
                        if f'sched_ext: BPF scheduler "{scheduler_name}_' in line and "enabled" in line:
                            print(f"Scheduler {scheduler_name} enabled successfully")
                            time.sleep(1)  # Give it a moment to fully initialize
                            return

        # Timeout occurred - print debug information
        self._print_debug_info(scheduler_name)
        raise RuntimeError(f"Timeout waiting for {scheduler_name} scheduler to be enabled")

    def _print_debug_info(self, scheduler_name: str) -> None:
        """Print debug information when scheduler fails to start."""
        print("\n" + "=" * 60)
        print("SCHEDULER STARTUP TIMEOUT - DEBUG INFORMATION")
        print("=" * 60)

        print("\nLast 10 dmesg lines:")
        if self.dmesg_lines:
            for line in self.dmesg_lines:
                print(f"  {line}")
        else:
            print("  (no dmesg output captured)")

        if self.scheduler_proc:
            print("\nScheduler process stdout (last 10 lines):")
            if self.scheduler_proc.stdout:
                try:
                    # Try to read any pending output
                    import fcntl
                    import os
                    fd = self.scheduler_proc.stdout.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                    stdout_lines = []
                    try:
                        while True:
                            line = self.scheduler_proc.stdout.readline()
                            if not line:
                                break
                            stdout_lines.append(line.strip())
                    except:
                        pass

                    if stdout_lines:
                        for line in stdout_lines[-10:]:
                            print(f"  {line}")
                    else:
                        print("  (no stdout output)")
                except Exception as e:
                    print(f"  (failed to read stdout: {e})")
            else:
                print("  (stdout not captured)")

            print("\nScheduler process stderr (last 10 lines):")
            if self.scheduler_proc.stderr:
                try:
                    # Try to read any pending output
                    import fcntl
                    import os
                    fd = self.scheduler_proc.stderr.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                    stderr_lines = []
                    try:
                        while True:
                            line = self.scheduler_proc.stderr.readline()
                            if not line:
                                break
                            stderr_lines.append(line.strip())
                    except:
                        pass

                    if stderr_lines:
                        for line in stderr_lines[-10:]:
                            print(f"  {line}")
                    else:
                        print("  (no stderr output)")
                except Exception as e:
                    print(f"  (failed to read stderr: {e})")
            else:
                print("  (stderr not captured)")

            print("\nScheduler process status:")
            if self.scheduler_proc.poll() is None:
                print("  Process is still running")
            else:
                print(f"  Process exited with code: {self.scheduler_proc.poll()}")

        print("=" * 60 + "\n")

    def _stop_process(self, proc: subprocess.Popen[str]) -> None:
        """Stop a process gracefully, with fallback to kill.

        Sends SIGINT first (like Ctrl-C) to allow the scheduler to cleanly
        unload itself from the kernel, then falls back to SIGTERM and SIGKILL.
        """
        import signal

        # First try SIGINT (like Ctrl-C) - this allows scheduler to unload cleanly
        print("  Sending SIGINT (Ctrl-C equivalent)...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
            print("  Process stopped gracefully with SIGINT")
            return
        except subprocess.TimeoutExpired:
            print("  Process didn't respond to SIGINT, trying SIGTERM...")

        # Fall back to SIGTERM
        proc.terminate()
        try:
            proc.wait(timeout=5)
            print("  Process stopped with SIGTERM")
            return
        except subprocess.TimeoutExpired:
            print("  Process didn't terminate gracefully, killing...")

        # Last resort: SIGKILL
        proc.kill()
        try:
            proc.wait(timeout=5)
            print("  Process killed successfully")
        except subprocess.TimeoutExpired:
            print("  Warning: Process may still be running")

    def _cleanup_dmesg(self) -> None:
        """Clean up dmesg monitoring process."""
        if self.dmesg_proc:
            self.dmesg_proc.terminate()
            try:
                self.dmesg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dmesg_proc.kill()
                try:
                    self.dmesg_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            self.dmesg_proc = None
