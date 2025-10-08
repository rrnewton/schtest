#!/usr/bin/env python3
"""
Parse lstopo XML output and represent CPU topology as strongly typed Python classes.

This script parses the XML output from `lstopo --output-format xml` or `hwloc-ls --output-format xml` 
and creates a hierarchical representation of the CPU topology.
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ProcessingUnit:
    """Represents a Processing Unit (hardware thread/logical CPU)."""

    os_index: int  # Physical CPU number (P#0, P#88, etc.)
    gp_index: int  # Global position index
    cpuset: str  # CPU affinity mask

    def __str__(self) -> str:
        return f"PU L#{self.gp_index} (P#{self.os_index})"


@dataclass
class Core:
    """Represents a physical CPU core."""

    os_index: int
    gp_index: int
    cpuset: str
    processing_units: List[ProcessingUnit] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Core L#{self.os_index}"

    def get_cpu_numbers(self) -> List[int]:
        """Get list of CPU numbers (os_index) for all PUs in this core."""
        return [pu.os_index for pu in self.processing_units]


@dataclass
class L2Cache:
    """Represents an L2 cache."""

    gp_index: int
    cache_size: int
    depth: int
    cache_linesize: int
    cache_associativity: int
    cpuset: str
    cores: List[Core] = field(default_factory=list)

    def __str__(self) -> str:
        size_kb = self.cache_size // 1024
        return f"L2 L#{self.gp_index} ({size_kb}KB)"


@dataclass
class L3Cache:
    """Represents an L3 cache."""

    gp_index: int
    cache_size: int
    depth: int
    cache_linesize: int
    cache_associativity: int
    cpuset: str
    cores: List[Core] = field(default_factory=list)
    l2_caches: List[L2Cache] = field(default_factory=list)

    def __str__(self) -> str:
        size_mb = self.cache_size // (1024 * 1024)
        return f"L3 L#{self.gp_index} ({size_mb}MB)"

    def get_cpu_numbers(self) -> List[int]:
        """Get list of CPU numbers for all PUs under this L3 cache."""
        cpu_numbers = []
        for core in self.cores:
            cpu_numbers.extend(core.get_cpu_numbers())
        for l2 in self.l2_caches:
            for core in l2.cores:
                cpu_numbers.extend(core.get_cpu_numbers())
        return sorted(cpu_numbers)


@dataclass
class Die:
    """Represents a CPU die."""

    os_index: int
    gp_index: int
    cpuset: str
    l3_caches: List[L3Cache] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Die L#{self.os_index}"

    def get_cpu_numbers(self) -> List[int]:
        """Get list of CPU numbers for all PUs under this die."""
        cpu_numbers = []
        for l3 in self.l3_caches:
            cpu_numbers.extend(l3.get_cpu_numbers())
        return sorted(cpu_numbers)


@dataclass
class NUMANode:
    """Represents a NUMA node."""

    os_index: int
    gp_index: int
    cpuset: str
    local_memory: int  # Memory in bytes

    def __str__(self) -> str:
        memory_gb = self.local_memory // (1024**3)
        return f"NUMANode L#{self.os_index} (P#{self.os_index} {memory_gb}GB)"


@dataclass
class Package:
    """Represents a CPU package (socket)."""

    os_index: int
    gp_index: int
    cpuset: str
    cpu_vendor: str
    cpu_model: str
    numa_nodes: List[NUMANode] = field(default_factory=list)
    dies: List[Die] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Package L#{self.os_index}"

    def get_cpu_numbers(self) -> List[int]:
        """Get list of CPU numbers for all PUs under this package."""
        cpu_numbers = []
        for die in self.dies:
            cpu_numbers.extend(die.get_cpu_numbers())
        return sorted(cpu_numbers)


@dataclass
class Machine:
    """Represents the entire machine topology."""

    os_index: int
    gp_index: int
    cpuset: str
    total_memory: int = 0  # Total memory across all NUMA nodes
    packages: List[Package] = field(default_factory=list)

    def __str__(self) -> str:
        memory_gb = self.total_memory // (1024**3)
        return f"Machine ({memory_gb}GB total)"

    def get_cpu_numbers(self) -> List[int]:
        """Get list of all CPU numbers (physical CPU IDs) in the machine."""
        cpu_numbers = []
        for package in self.packages:
            cpu_numbers.extend(package.get_cpu_numbers())
        return sorted(cpu_numbers)

    def split_hyperthreads(self) -> Tuple[List[int], List[int]]:
        """
        Split CPUs where hyperthreads within each core are in different partitions.

        Returns:
            Tuple of two lists of CPU numbers, where each core's hyperthreads
            are split between the two lists.

        Raises:
            ValueError: If the machine doesn't have hyperthreading/SMT.
        """
        partition_a = []
        partition_b = []

        # Collect all cores across all packages and dies
        all_cores = []
        for package in self.packages:
            for die in package.dies:
                for l3 in die.l3_caches:
                    all_cores.extend(l3.cores)
                    for l2 in l3.l2_caches:
                        all_cores.extend(l2.cores)

        if not all_cores:
            raise ValueError("No cores found in topology")

        # Check if we have hyperthreading
        has_hyperthreading = any(len(core.processing_units) > 1 for core in all_cores)
        if not has_hyperthreading:
            raise ValueError("Machine does not have hyperthreading/SMT")

        # Split hyperthreads across partitions
        for core in all_cores:
            if len(core.processing_units) == 0:
                continue
            elif len(core.processing_units) == 1:
                # Single thread core - put in partition A
                partition_a.append(core.processing_units[0].os_index)
            else:
                # Multiple threads - split them
                sorted_pus = sorted(core.processing_units, key=lambda pu: pu.os_index)
                for i, pu in enumerate(sorted_pus):
                    if i % 2 == 0:
                        partition_a.append(pu.os_index)
                    else:
                        partition_b.append(pu.os_index)

        return sorted(partition_a), sorted(partition_b)

    def split_dies(self) -> Tuple[List[int], List[int]]:
        """
        Split the machine in half at the Die level.

        Returns:
            Tuple of two lists of CPU numbers, where dies are split evenly
            between the two partitions.

        Raises:
            ValueError: If there are an odd number of dies.
        """
        partition_a = []
        partition_b = []

        # Collect all dies across all packages
        all_dies = []
        for package in self.packages:
            all_dies.extend(package.dies)

        if len(all_dies) % 2 != 0:
            raise ValueError(
                f"Cannot split dies evenly: found {len(all_dies)} dies (odd number)"
            )

        if not all_dies:
            raise ValueError("No dies found in topology")

        # Sort dies by os_index for consistent partitioning
        sorted_dies = sorted(all_dies, key=lambda die: die.os_index)

        # Split dies in half
        mid_point = len(sorted_dies) // 2

        for die in sorted_dies[:mid_point]:
            partition_a.extend(die.get_cpu_numbers())

        for die in sorted_dies[mid_point:]:
            partition_b.extend(die.get_cpu_numbers())

        return sorted(partition_a), sorted(partition_b)


class LstopoParser:
    """Parser for lstopo XML output."""

    def __init__(self):
        self.machine: Optional[Machine] = None

    def _run_command(self, cmd: List[str]) -> str:
        """Run a command and return its output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\nError: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {cmd[0]}")

    def _get_topology_xml(self) -> str:
        """Get topology XML from lstopo or hwloc-ls."""
        # Try lstopo first
        try:
            return self._run_command(
                ["lstopo", "--output-format", "xml", "--no-io", "--no-bridges"]
            )
        except RuntimeError:
            pass

        # Fall back to hwloc-ls
        try:
            return self._run_command(
                ["hwloc-ls", "--output-format", "xml", "--no-io", "--no-bridges"]
            )
        except RuntimeError:
            pass

        raise RuntimeError("Neither 'lstopo' nor 'hwloc-ls' commands are available")

    def _parse_processing_unit(self, pu_elem: ET.Element) -> ProcessingUnit:
        """Parse a Processing Unit (PU) element."""
        return ProcessingUnit(
            os_index=int(pu_elem.get("os_index", 0)),
            gp_index=int(pu_elem.get("gp_index", 0)),
            cpuset=pu_elem.get("cpuset", ""),
        )

    def _parse_core(self, core_elem: ET.Element) -> Core:
        """Parse a Core element."""
        core = Core(
            os_index=int(core_elem.get("os_index", 0)),
            gp_index=int(core_elem.get("gp_index", 0)),
            cpuset=core_elem.get("cpuset", ""),
        )

        # Parse processing units
        for pu_elem in core_elem.findall('.//object[@type="PU"]'):
            core.processing_units.append(self._parse_processing_unit(pu_elem))

        return core

    def _parse_l2_cache(self, l2_elem: ET.Element) -> L2Cache:
        """Parse an L2 cache element."""
        l2 = L2Cache(
            gp_index=int(l2_elem.get("gp_index", 0)),
            cache_size=int(l2_elem.get("cache_size", 0)),
            depth=int(l2_elem.get("depth", 2)),
            cache_linesize=int(l2_elem.get("cache_linesize", 64)),
            cache_associativity=int(l2_elem.get("cache_associativity", 0)),
            cpuset=l2_elem.get("cpuset", ""),
        )

        # Parse cores directly under L2
        for core_elem in l2_elem.findall('./object[@type="Core"]'):
            l2.cores.append(self._parse_core(core_elem))

        # If no cores found directly, search recursively through L1 caches
        # This handles cases where cores are nested under L1/L1i caches
        if not l2.cores:
            for core_elem in l2_elem.findall('.//object[@type="Core"]'):
                l2.cores.append(self._parse_core(core_elem))

        return l2

    def _parse_l3_cache(self, l3_elem: ET.Element) -> L3Cache:
        """Parse an L3 cache element."""
        l3 = L3Cache(
            gp_index=int(l3_elem.get("gp_index", 0)),
            cache_size=int(l3_elem.get("cache_size", 0)),
            depth=int(l3_elem.get("depth", 3)),
            cache_linesize=int(l3_elem.get("cache_linesize", 64)),
            cache_associativity=int(l3_elem.get("cache_associativity", 0)),
            cpuset=l3_elem.get("cpuset", ""),
        )

        # Parse cores directly under L3 (for simplified topologies)
        for core_elem in l3_elem.findall('./object[@type="Core"]'):
            l3.cores.append(self._parse_core(core_elem))

        # Parse L2 caches under L3 (for detailed topologies)
        for l2_elem in l3_elem.findall('./object[@type="L2Cache"]'):
            l3.l2_caches.append(self._parse_l2_cache(l2_elem))

        # If no cores found yet, search recursively through all cache levels
        # This handles cases where cores are nested under L1/L1i caches
        if not l3.cores and not l3.l2_caches:
            for core_elem in l3_elem.findall('.//object[@type="Core"]'):
                l3.cores.append(self._parse_core(core_elem))

        return l3

    def _parse_die(self, die_elem: ET.Element) -> Die:
        """Parse a Die element."""
        die = Die(
            os_index=int(die_elem.get("os_index", 0)),
            gp_index=int(die_elem.get("gp_index", 0)),
            cpuset=die_elem.get("cpuset", ""),
        )

        # Parse L3 caches (direct children only)
        for l3_elem in die_elem.findall('./object[@type="L3Cache"]'):
            die.l3_caches.append(self._parse_l3_cache(l3_elem))

        return die

    def _parse_numa_node(self, numa_elem: ET.Element) -> NUMANode:
        """Parse a NUMA node element."""
        return NUMANode(
            os_index=int(numa_elem.get("os_index", 0)),
            gp_index=int(numa_elem.get("gp_index", 0)),
            cpuset=numa_elem.get("cpuset", ""),
            local_memory=int(numa_elem.get("local_memory", 0)),
        )

    def _parse_package(self, package_elem: ET.Element) -> Package:
        """Parse a Package element."""
        # Extract CPU info
        cpu_vendor = ""
        cpu_model = ""
        for info_elem in package_elem.findall("./info"):
            name = info_elem.get("name", "")
            value = info_elem.get("value", "")
            if name == "CPUVendor":
                cpu_vendor = value
            elif name == "CPUModel":
                cpu_model = value.strip()

        package = Package(
            os_index=int(package_elem.get("os_index", 0)),
            gp_index=int(package_elem.get("gp_index", 0)),
            cpuset=package_elem.get("cpuset", ""),
            cpu_vendor=cpu_vendor,
            cpu_model=cpu_model,
        )

        # Parse NUMA nodes
        for numa_elem in package_elem.findall('./object[@type="NUMANode"]'):
            package.numa_nodes.append(self._parse_numa_node(numa_elem))

        # Parse dies
        for die_elem in package_elem.findall('./object[@type="Die"]'):
            package.dies.append(self._parse_die(die_elem))

        return package

    def _parse_machine(self, machine_elem: ET.Element) -> Machine:
        """Parse the Machine element."""
        machine = Machine(
            os_index=int(machine_elem.get("os_index", 0)),
            gp_index=int(machine_elem.get("gp_index", 0)),
            cpuset=machine_elem.get("cpuset", ""),
        )

        # Parse packages
        for package_elem in machine_elem.findall('./object[@type="Package"]'):
            package = self._parse_package(package_elem)
            machine.packages.append(package)

            # Sum up memory from NUMA nodes
            for numa_node in package.numa_nodes:
                machine.total_memory += numa_node.local_memory

        return machine

    def parse_xml(self, xml_content: str) -> Machine:
        """Parse XML content and return Machine topology."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML content: {e}")

        # Find the machine element
        machine_elem = root.find('./object[@type="Machine"]')
        if machine_elem is None:
            raise ValueError("No Machine object found in topology XML")

        self.machine = self._parse_machine(machine_elem)
        return self.machine

    def parse_topology(self) -> Machine:
        """Parse topology from lstopo/hwloc-ls and return Machine."""
        xml_content = self._get_topology_xml()
        return self.parse_xml(xml_content)


def parse_topology() -> Machine:
    """
    Primary function to parse CPU topology.

    Returns:
        Machine object representing the complete CPU topology.
    """
    parser = LstopoParser()
    return parser.parse_topology()


def main():
    """Main function for CLI usage."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        return

    try:
        machine = parse_topology()

        print(machine)
        for package in machine.packages:
            print(f"  {package}")
            for numa_node in package.numa_nodes:
                print(f"    {numa_node}")
            for die in package.dies:
                print(f"    {die}")
                for l3 in die.l3_caches:
                    print(f"      {l3}")
                    for core in l3.cores:
                        print(f"        {core}")
                        for pu in core.processing_units:
                            print(f"          {pu}")

        print(f"\nAll CPU numbers: {machine.get_cpu_numbers()}")

        try:
            a, b = machine.split_hyperthreads()
            print(f"\nHyperthread split A: {a}")
            print(f"Hyperthread split B: {b}")
        except ValueError as e:
            print(f"\nHyperthread split error: {e}")

        try:
            a, b = machine.split_dies()
            print(f"\nDie split A: {a}")
            print(f"Die split B: {b}")
        except ValueError as e:
            print(f"\nDie split error: {e}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
