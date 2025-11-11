#!/usr/bin/env python3
"""Emergency CPU frequency restore script."""

import os
import sys
from pathlib import Path

CPUFREQ_BASE = Path("/sys/devices/system/cpu")

def restore_all_cpus():
    """Restore all CPUs to unrestricted state."""
    if os.geteuid() != 0:
        print("ERROR: Must run as root (use sudo)")
        sys.exit(1)

    cpus = sorted([d for d in CPUFREQ_BASE.glob("cpu[0-9]*") if (d / "cpufreq").exists()])

    print(f"Restoring {len(cpus)} CPUs to unrestricted state...")

    for cpu_dir in cpus:
        cpu_id = cpu_dir.name
        cpufreq_dir = cpu_dir / "cpufreq"

        try:
            # Read hardware limits
            cpuinfo_min = (cpufreq_dir / "cpuinfo_min_freq").read_text().strip()
            cpuinfo_max = (cpufreq_dir / "cpuinfo_max_freq").read_text().strip()

            # Set governor to schedutil (or performance as fallback)
            try:
                (cpufreq_dir / "scaling_governor").write_text("schedutil\n")
                governor = "schedutil"
            except:
                (cpufreq_dir / "scaling_governor").write_text("performance\n")
                governor = "performance"

            # Set min/max to hardware limits (unrestricted)
            (cpufreq_dir / "scaling_min_freq").write_text(cpuinfo_min + "\n")
            (cpufreq_dir / "scaling_max_freq").write_text(cpuinfo_max + "\n")

            print(f"  {cpu_id}: {governor}, [{cpuinfo_min}, {cpuinfo_max}]")

        except Exception as e:
            print(f"  Warning: Failed to restore {cpu_id}: {e}")

    print("Done!")

if __name__ == "__main__":
    restore_all_cpus()
