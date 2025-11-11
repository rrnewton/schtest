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


class DmesgMonitor:
    """Monitor dmesg for scheduler-related kernel messages."""

    def __init__(self, scheduler_name: str) -> None:
        """Start dmesg monitoring for the given scheduler.

        Args:
            scheduler_name: Name of the scheduler to monitor (e.g., "lavd")
        """
        self.scheduler_name = scheduler_name
        self.dmesg_lines: List[str] = []
        self.dmesg_proc: Optional[subprocess.Popen[str]] = None

        # Start dmesg monitoring with process group
        # Using start_new_session=True makes the process a session leader
        # so we can kill the entire process tree
        self.dmesg_proc = subprocess.Popen(
            ["sudo", "dmesg", "-W"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True
        )

        if self.dmesg_proc.stdout is None:
            self.cleanup()
            raise RuntimeError("Failed to start dmesg monitoring")

        print(f"Started dmesg monitoring for {scheduler_name} scheduler...")

    def wait_for_enabled(self, timeout: float = 30.0) -> None:
        """Wait for scheduler enabled message in dmesg.

        Args:
            timeout: Maximum time to wait (seconds)

        Raises:
            RuntimeError: If timeout occurs or dmesg process dies
        """
        print(f"Waiting for {self.scheduler_name} scheduler to be enabled...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.dmesg_proc and self.dmesg_proc.poll() is not None:
                # Process died - capture stderr for diagnostics
                stderr_output = ""
                if self.dmesg_proc.stderr:
                    stderr_output = self.dmesg_proc.stderr.read().strip()

                error_msg = f"dmesg monitoring process died (exit code: {self.dmesg_proc.returncode})"
                if stderr_output:
                    error_msg += f"\nStderr: {stderr_output}"
                    # Check for common issues
                    if "no new privileges" in stderr_output.lower():
                        error_msg += "\n\nThis is likely due to running in a sandboxed environment."
                        error_msg += "\nTry running without sandbox restrictions or use 'dmesg' without sudo."

                raise RuntimeError(error_msg)

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
                        if f'sched_ext: BPF scheduler "{self.scheduler_name}_' in line and "enabled" in line:
                            print(f"Scheduler {self.scheduler_name} enabled successfully")
                            time.sleep(1)  # Give it a moment to fully initialize
                            return

        # Timeout occurred
        raise RuntimeError(f"Timeout waiting for {self.scheduler_name} scheduler to be enabled")

    def wait_for_disabled(self, timeout: float = 15.0) -> None:
        """Wait for scheduler disabled message in dmesg.

        Args:
            timeout: Maximum time to wait (seconds)

        Raises:
            RuntimeError: If timeout occurs waiting for kernel disable confirmation
        """
        print(f"Waiting for {self.scheduler_name} scheduler to be disabled in kernel...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.dmesg_proc and self.dmesg_proc.poll() is not None:
                # dmesg process died - warn but don't fail
                print("  Warning: dmesg monitoring process died during shutdown")
                return

            # Read with timeout to avoid blocking forever
            if self.dmesg_proc and self.dmesg_proc.stdout:
                ready, _, _ = select.select([self.dmesg_proc.stdout], [], [], 1.0)

                if ready:
                    line = self.dmesg_proc.stdout.readline()
                    if line:
                        stripped = line.strip()
                        print(f"dmesg: {stripped}")

                        # Look for scheduler disabled message
                        if f'sched_ext: BPF scheduler "{self.scheduler_name}_' in line and "disabled" in line:
                            print(f"Scheduler {self.scheduler_name} disabled successfully in kernel")
                            time.sleep(0.5)  # Give it a moment to fully clean up
                            return

                        # Also look for "unregistered" message
                        if "sched_ext" in line and "unregistered from user space" in line:
                            print(f"Scheduler {self.scheduler_name} disabled successfully in kernel")
                            time.sleep(0.5)
                            return

        # Timeout occurred - this is a fatal error
        raise RuntimeError(
            f"Timeout waiting for {self.scheduler_name} scheduler disable confirmation. "
            f"Scheduler process was stopped, but kernel disable message not seen in dmesg within {timeout}s"
        )

    def get_recent_lines(self) -> List[str]:
        """Get the last 10 lines captured from dmesg."""
        return list(self.dmesg_lines)

    def cleanup(self) -> None:
        """Clean up dmesg monitoring process and its children."""
        if self.dmesg_proc:
            import os
            import signal

            # Kill the entire process group (sudo + dmesg)
            try:
                if self.dmesg_proc.poll() is None:  # Process still running
                    os.killpg(os.getpgid(self.dmesg_proc.pid), signal.SIGTERM)
                    try:
                        self.dmesg_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(self.dmesg_proc.pid), signal.SIGKILL)
                        try:
                            self.dmesg_proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
            except (ProcessLookupError, PermissionError):
                # Process or group already gone
                pass

            self.dmesg_proc = None


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
        self.dmesg_monitor: Optional[DmesgMonitor] = None

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
        self.dmesg_monitor = DmesgMonitor(scheduler_name)

        try:
            # Start the scheduler process with sudo
            # Using start_new_session=True makes the process a session leader
            # so we can kill the entire process tree (sudo + actual scheduler)
            self.scheduler_proc = subprocess.Popen(
                ["sudo", str(self.scheduler_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )

            # Wait for scheduler to be enabled
            self.dmesg_monitor.wait_for_enabled(timeout)

            return self.scheduler_proc

        except RuntimeError:
            # Print debug info on timeout
            self._print_debug_info()
            # Clean up on failure
            if self.dmesg_monitor:
                self.dmesg_monitor.cleanup()
                self.dmesg_monitor = None
            if self.scheduler_proc:
                self._stop_process(self.scheduler_proc)
                self.scheduler_proc = None
            raise
        except Exception:
            # Clean up on other failures
            if self.dmesg_monitor:
                self.dmesg_monitor.cleanup()
                self.dmesg_monitor = None
            if self.scheduler_proc:
                self._stop_process(self.scheduler_proc)
                self.scheduler_proc = None
            raise

    def stop(self) -> None:
        """Stop the scheduler process and wait for kernel to confirm it's disabled.

        Sends SIGINT to allow the scheduler to cleanly unload itself from
        the kernel (same as Ctrl-C behavior), then waits for dmesg confirmation.
        """
        if self.scheduler_proc is None:
            return

        print("Stopping scheduler...")

        # Stop the scheduler process (sends SIGINT first, then SIGTERM, then SIGKILL)
        self._stop_process(self.scheduler_proc)
        self.scheduler_proc = None

        # Wait for kernel to confirm scheduler is disabled
        if self.dmesg_monitor:
            self.dmesg_monitor.wait_for_disabled()
            self.dmesg_monitor.cleanup()
            self.dmesg_monitor = None

        print("Scheduler stopped")

    def _print_debug_info(self) -> None:
        """Print debug information when scheduler fails to start."""
        print("\n" + "=" * 60)
        print("SCHEDULER STARTUP TIMEOUT - DEBUG INFORMATION")
        print("=" * 60)

        print("\nLast 10 dmesg lines:")
        if self.dmesg_monitor:
            dmesg_lines = self.dmesg_monitor.get_recent_lines()
            if dmesg_lines:
                for line in dmesg_lines:
                    print(f"  {line}")
            else:
                print("  (no dmesg output captured)")
        else:
            print("  (no dmesg monitor active)")

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
        """Stop a process and its children gracefully, with fallback to kill.

        Sends SIGINT first (like Ctrl-C) to the process group to allow the scheduler
        to cleanly unload itself from the kernel, then falls back to SIGTERM and SIGKILL.

        Since we start processes with sudo, we need to kill the entire process group
        (sudo + the actual scheduler) not just the sudo wrapper.
        """
        import signal
        import os

        if proc.poll() is not None:
            # Process already exited
            print("  Process already exited")
            return

        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            print("  Process or process group not found")
            return

        # First try SIGINT (like Ctrl-C) to the process group - this allows scheduler to unload cleanly
        print("  Sending SIGINT to process group (Ctrl-C equivalent)...")
        try:
            os.killpg(pgid, signal.SIGINT)
            try:
                proc.wait(timeout=10)
                print("  Process stopped gracefully with SIGINT")
                return
            except subprocess.TimeoutExpired:
                print("  Process didn't respond to SIGINT, trying SIGTERM...")
        except (ProcessLookupError, PermissionError):
            print("  Process group already gone")
            return

        # Fall back to SIGTERM for the process group
        try:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
                print("  Process stopped with SIGTERM")
                return
            except subprocess.TimeoutExpired:
                print("  Process didn't terminate gracefully, killing...")
        except (ProcessLookupError, PermissionError):
            print("  Process group already gone")
            return

        # Last resort: SIGKILL to process group
        try:
            os.killpg(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=5)
                print("  Process killed successfully")
            except subprocess.TimeoutExpired:
                print("  Warning: Process may still be running")
        except (ProcessLookupError, PermissionError):
            print("  Process group already gone")
