"""Tests for VCF reader."""
import pytest

from pysingerarg.io.vcf_reader import read_vcf_phased, read_vcf_haploid


class TestReadVcfPhased:
    def test_loads_nodes(self, small_vcf_path):
        nodes, seq_len = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=1000)
        # 5 samples × 2 haplotypes = 10 nodes
        assert len(nodes) == 10

    def test_sequence_length(self, small_vcf_path):
        _, seq_len = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=1000)
        assert seq_len == 1000.0

    def test_mutations_present(self, small_vcf_path):
        nodes, _ = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=1000)
        # At least some nodes should have mutations
        any_mut = any(
            any(pos >= 0 for pos in n.mutation_sites.keys())
            for n in nodes
        )
        assert any_mut

    def test_region_filtering(self, small_vcf_path):
        # Only load positions [0, 300)
        nodes_full, _ = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=1000)
        nodes_sub, _ = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=300)
        # Subset should have fewer or equal mutations
        total_full = sum(sum(1 for p in n.mutation_sites.keys() if p >= 0) for n in nodes_full)
        total_sub = sum(sum(1 for p in n.mutation_sites.keys() if p >= 0) for n in nodes_sub)
        assert total_sub <= total_full

    def test_node_indices(self, small_vcf_path):
        nodes, _ = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=1000)
        indices = [n.index for n in nodes]
        assert sorted(indices) == list(range(len(nodes)))

    def test_all_nodes_at_time_zero(self, small_vcf_path):
        nodes, _ = read_vcf_phased(small_vcf_path, start_pos=0, end_pos=1000)
        assert all(n.time == 0.0 for n in nodes)
