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

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from parse_topo import Machine, parse_topology


@pytest.fixture(scope="session")
def machine():
    """Fixture to parse topology once for all tests."""
    print("Parsing CPU topology...")
    parsed_machine = parse_topology()
    print(f"Topology parsed: {parsed_machine}")
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

    def test_split_dies_properties(self, machine):
        """Test split_dies satisfies the required properties."""
        try:
            die_a, die_b = machine.split_dies()
        except ValueError as e:
            pytest.skip(f"Die split not possible: {e}")

        all_cpus = set(machine.get_cpu_numbers())
        set_a = set(die_a)
        set_b = set(die_b)

        # Property 1: Note - die splits may NOT have equal numbers since dies can have different CPU counts
        # We just verify both partitions have some CPUs
        assert len(die_a) > 0, "Die partition A should have CPUs"
        assert len(die_b) > 0, "Die partition B should have CPUs"

        # Property 2: Disjoint sets (no CPU appears in both partitions)
        intersection = set_a & set_b
        assert len(intersection) == 0, (
            f"Die splits should be disjoint, found overlap: {intersection}"
        )

        # Property 3: Union equals complete set
        union = set_a | set_b
        assert union == all_cpus, (
            f"Die splits should union to complete set. "
            f"Missing: {all_cpus - union}, Extra: {union - all_cpus}"
        )

        # Additional validation: splits should be sorted
        assert die_a == sorted(die_a), "Partition A should be sorted"
        assert die_b == sorted(die_b), "Partition B should be sorted"

        print(
            f"✓ Die split validation passed: {len(die_a)} + {len(die_b)} = {len(all_cpus)} CPUs"
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

        # Test die split consistency
        try:
            die_a1, die_b1 = machine.split_dies()
            die_a2, die_b2 = machine.split_dies()

            assert die_a1 == die_a2, "Multiple die splits should return same partition A"
            assert die_b1 == die_b2, "Multiple die splits should return same partition B"
            print("✓ Die split consistency verified")
        except ValueError:
            print("⚠ Skipping die consistency test (not available)")


# Optional: provide backward compatibility if someone runs this file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
