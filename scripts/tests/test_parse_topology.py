#!/usr/bin/env python3
"""
pytest-style unit tests for the parse_topology function and CPU topology parsing.

Modern CI/CD-friendly testing approach using pytest.

Tests the parsing of CPU topology and the split functions to ensure:
1. Splits have equal number of cores (when expected)
2. Splits are disjoint (no overlap)
3. Splits union back to the complete set
"""

import os
import sys
import pytest
from pathlib import Path
from typing import List

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from parse_topo import Machine, LstopoParser, parse_topology


def get_test_topologies() -> List[tuple[str, Path]]:
    """Get list of test topology XML files."""
    topos_dir = Path(__file__).parent / "topos"
    if not topos_dir.exists():
        return []

    xml_files = sorted(topos_dir.glob("*.xml"))
    return [(xml_file.stem, xml_file) for xml_file in xml_files]


@pytest.fixture(scope="session", params=["native"] + [name for name, _ in get_test_topologies()])
def machine(request):
    """Fixture to parse topology - either native or from test XML files."""
    topology_name = request.param

    if topology_name == "native":
        print("\nParsing native CPU topology...")
        parsed_machine = parse_topology()
    else:
        # Find the XML file for this topology
        xml_path = None
        for name, path in get_test_topologies():
            if name == topology_name:
                xml_path = path
                break

        if xml_path is None:
            pytest.skip(f"Topology file not found: {topology_name}")

        print(f"\nParsing topology from {xml_path.name}...")
        with open(xml_path, 'r') as f:
            xml_content = f.read()

        parser = LstopoParser()
        parsed_machine = parser.parse_xml(xml_content)

    print(f"Topology parsed: {parsed_machine} (source: {topology_name})")
    return parsed_machine


class TestParseTopology:
    """Test cases for CPU topology parsing and splitting."""

    def test_parse_topology_basic(self, machine):
        """Test that parse_topology returns a valid Machine object."""
        # Basic validation
        assert isinstance(machine, Machine)
        assert len(machine.packages) > 0, "Machine should have at least one package"
        assert machine.total_memory > 0, "Machine should have memory"

        # CPU validation
        cpu_numbers = machine.get_cpu_numbers()
        assert len(cpu_numbers) > 0, "Machine should have CPUs"
        assert len(cpu_numbers) == len(set(cpu_numbers)), "CPU numbers should be unique"
        assert cpu_numbers == sorted(cpu_numbers), "CPU numbers should be sorted"

        print(
            f"✓ Basic validation passed: {len(cpu_numbers)} CPUs, {machine.total_memory // (1024**3)}GB memory"
        )

    def test_split_hyperthreads_properties(self, machine):
        """Test split_hyperthreads satisfies the required properties."""
        try:
            ht_a, ht_b = machine.split_hyperthreads()
        except ValueError as e:
            pytest.skip(f"Hyperthreading not available: {e}")

        all_cpus = set(machine.get_cpu_numbers())
        set_a = set(ht_a)
        set_b = set(ht_b)

        # Property 1: Equal number of cores (each physical core contributes one thread to each partition)
        assert len(ht_a) == len(ht_b), (
            f"Hyperthread splits should have equal sizes: {len(ht_a)} vs {len(ht_b)}"
        )

        # Property 2: Disjoint sets (no CPU appears in both partitions)
        intersection = set_a & set_b
        assert len(intersection) == 0, (
            f"Hyperthread splits should be disjoint, found overlap: {intersection}"
        )

        # Property 3: Union equals complete set
        union = set_a | set_b
        assert union == all_cpus, (
            f"Hyperthread splits should union to complete set. "
            f"Missing: {all_cpus - union}, Extra: {union - all_cpus}"
        )

        # Additional validation: splits should be sorted
        assert ht_a == sorted(ht_a), "Partition A should be sorted"
        assert ht_b == sorted(ht_b), "Partition B should be sorted"

        print(
            f"✓ Hyperthread split validation passed: {len(ht_a)} + {len(ht_b)} = {len(all_cpus)} CPUs"
        )

    def test_split_l3s_properties(self, machine):
        """Test split_l3s satisfies the required properties."""
        try:
            l3_a, l3_b = machine.split_l3s()
        except ValueError as e:
            pytest.skip(f"L3 split not possible: {e}")

        all_cpus = set(machine.get_cpu_numbers())
        set_a = set(l3_a)
        set_b = set(l3_b)

        # Property 1: Note - L3 splits may NOT have equal numbers since L3s can have different CPU counts
        # We just verify both partitions have some CPUs
        assert len(l3_a) > 0, "L3 partition A should have CPUs"
        assert len(l3_b) > 0, "L3 partition B should have CPUs"

        # Property 2: Disjoint sets (no CPU appears in both partitions)
        intersection = set_a & set_b
        assert len(intersection) == 0, (
            f"L3 splits should be disjoint, found overlap: {intersection}"
        )

        # Property 3: Union equals complete set
        union = set_a | set_b
        assert union == all_cpus, (
            f"L3 splits should union to complete set. "
            f"Missing: {all_cpus - union}, Extra: {union - all_cpus}"
        )

        # Additional validation: splits should be sorted
        assert l3_a == sorted(l3_a), "Partition A should be sorted"
        assert l3_b == sorted(l3_b), "Partition B should be sorted"

        print(
            f"✓ L3 split validation passed: {len(l3_a)} + {len(l3_b)} = {len(all_cpus)} CPUs"
        )

    def test_split_physical_properties(self, machine):
        """Test split_physical satisfies the required properties.

        This test is stricter than split_l3s because split_physical should ALWAYS succeed
        by trying multiple hierarchy levels (Package, NUMANode, Die, L3, L2, Core, Hyperthread).
        """
        # Should never raise ValueError - split_physical tries multiple levels
        level, phys_a, phys_b = machine.split_physical()

        assert level is not None, "split_physical should return a valid level name"
        assert isinstance(level, str), "Level name should be a string"
        assert level in ["Package", "Die", "L3", "L2", "Core", "Hyperthread"], (
            f"Level '{level}' should be one of the hierarchy levels"
        )

        all_cpus = set(machine.get_cpu_numbers())
        set_a = set(phys_a)
        set_b = set(phys_b)

        # Property 1: Both partitions must have CPUs (stricter - no skip)
        assert len(phys_a) > 0, "Physical partition A must have CPUs"
        assert len(phys_b) > 0, "Physical partition B must have CPUs"

        # Property 2: Disjoint sets (no CPU appears in both partitions)
        intersection = set_a & set_b
        assert len(intersection) == 0, (
            f"Physical splits should be disjoint, found overlap: {intersection}"
        )

        # Property 3: Union equals complete set
        union = set_a | set_b
        assert union == all_cpus, (
            f"Physical splits should union to complete set. "
            f"Missing: {all_cpus - union}, Extra: {union - all_cpus}"
        )

        # Additional validation: splits should be sorted
        assert phys_a == sorted(phys_a), "Partition A should be sorted"
        assert phys_b == sorted(phys_b), "Partition B should be sorted"

        print(
            f"✓ Physical split validation passed at {level} level: "
            f"{len(phys_a)} + {len(phys_b)} = {len(all_cpus)} CPUs"
        )

    def test_split_hyperthreads_hyperthread_pattern(self, machine):
        """Test that hyperthread split actually separates hyperthreads within cores."""
        try:
            ht_a, ht_b = machine.split_hyperthreads()
        except ValueError as e:
            pytest.skip(f"Hyperthreading not available: {e}")

        # Collect all cores to verify hyperthreading pattern
        all_cores = []
        for package in machine.packages:
            for die in package.dies:
                for l3 in die.l3_caches:
                    all_cores.extend(l3.cores)
                    for l2 in l3.l2_caches:
                        all_cores.extend(l2.cores)

        set_a = set(ht_a)
        set_b = set(ht_b)

        # For each core with multiple processing units, verify they're split across partitions
        cores_with_multiple_pus = [
            core for core in all_cores if len(core.processing_units) > 1
        ]
        assert len(cores_with_multiple_pus) > 0, "Should have cores with multiple processing units"

        split_cores_correctly = 0
        for core in cores_with_multiple_pus:
            core_cpus = core.get_cpu_numbers()
            if len(core_cpus) >= 2:
                # Check that core's CPUs are split between partitions
                cpus_in_a = [cpu for cpu in core_cpus if cpu in set_a]
                cpus_in_b = [cpu for cpu in core_cpus if cpu in set_b]

                # For hyperthreading, we expect roughly equal split within each core
                if len(cpus_in_a) > 0 and len(cpus_in_b) > 0:
                    split_cores_correctly += 1

        # Verify that at least some cores had their hyperthreads split across partitions
        assert split_cores_correctly > 0, (
            "At least some cores should have their hyperthreads split across partitions"
        )

        print(
            f"✓ Hyperthread pattern validation passed: {split_cores_correctly} cores properly split"
        )

    def test_cpu_number_range_validity(self, machine):
        """Test that CPU numbers are in a valid range and contiguous or follow expected pattern."""
        cpu_numbers = machine.get_cpu_numbers()

        # Basic range validation
        assert all(cpu >= 0 for cpu in cpu_numbers), "All CPU numbers should be non-negative"

        min_cpu = min(cpu_numbers)
        max_cpu = max(cpu_numbers)

        # Log the range for debugging
        print(
            f"✓ CPU range validation: {len(cpu_numbers)} CPUs from {min_cpu} to {max_cpu}"
        )

        # Verify reasonable range (shouldn't have huge gaps)
        total_range = max_cpu - min_cpu + 1
        cpu_count = len(cpu_numbers)

        # Allow for some gaps but not excessive ones (e.g., hyperthreading often creates patterns like 0-87, 88-175)
        assert total_range <= cpu_count * 2, (
            f"CPU range ({total_range}) seems too sparse for {cpu_count} CPUs"
        )

    def test_topology_structure_validity(self, machine):
        """Test that the topology structure is reasonable."""
        # Count topology elements
        total_packages = len(machine.packages)
        total_dies = sum(len(pkg.dies) for pkg in machine.packages)
        total_l3_caches = sum(
            len(die.l3_caches) for pkg in machine.packages for die in pkg.dies
        )
        total_numa_nodes = sum(len(pkg.numa_nodes) for pkg in machine.packages)

        # Validation
        assert total_packages > 0, "Should have at least one package"
        # All packages should have at least one die (explicit or implicit)
        assert total_dies >= total_packages, "Should have at least as many dies as packages"
        assert total_l3_caches >= total_dies, "Should have at least as many L3 caches as dies"
        assert total_numa_nodes >= 0, "NUMA nodes count should be non-negative"

        print(
            f"✓ Topology structure: {total_packages} packages, {total_dies} dies, "
            f"{total_l3_caches} L3 caches, {total_numa_nodes} NUMA nodes"
        )


class TestSplitEdgeCases:
    """Test edge cases and error conditions for split functions."""

    def test_multiple_splits_consistency(self, machine):
        """Test that multiple calls to split functions return consistent results."""
        # Test hyperthread split consistency
        try:
            ht_a1, ht_b1 = machine.split_hyperthreads()
            ht_a2, ht_b2 = machine.split_hyperthreads()

            assert ht_a1 == ht_a2, "Multiple hyperthread splits should return same partition A"
            assert ht_b1 == ht_b2, "Multiple hyperthread splits should return same partition B"
            print("✓ Hyperthread split consistency verified")
        except ValueError:
            print("⚠ Skipping hyperthread consistency test (not available)")

        # Test L3 split consistency
        try:
            l3_a1, l3_b1 = machine.split_l3s()
            l3_a2, l3_b2 = machine.split_l3s()

            assert l3_a1 == l3_a2, "Multiple L3 splits should return same partition A"
            assert l3_b1 == l3_b2, "Multiple L3 splits should return same partition B"
            print("✓ L3 split consistency verified")
        except ValueError:
            print("⚠ Skipping L3 consistency test (not available)")

        # Test physical split consistency (should always succeed)
        level1, phys_a1, phys_b1 = machine.split_physical()
        level2, phys_a2, phys_b2 = machine.split_physical()

        assert level1 == level2, "Multiple physical splits should return same level"
        assert phys_a1 == phys_a2, "Multiple physical splits should return same partition A"
        assert phys_b1 == phys_b2, "Multiple physical splits should return same partition B"
        print(f"✓ Physical split consistency verified at {level1} level")


# Optional: provide backward compatibility if someone runs this file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
