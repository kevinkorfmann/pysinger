"""End-to-end integration test for the Sampler."""
import math
import pytest

from pysingerarg.sampler import Sampler


class TestSamplerIterativeStart:
    def test_loads_vcf(self, small_vcf_path):
        s = Sampler(Ne=10000.0, recomb_rate=1e-8, mut_rate=1e-8)
        s.load_vcf(small_vcf_path, start=0, end=1000)
        assert len(s.ordered_sample_nodes) == 10  # 5 ind × 2 haplotypes
        assert s.sequence_length == 1000.0

    def test_iterative_start_builds_arg(self, small_vcf_path):
        s = Sampler(Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4)
        s.set_seed(42)
        s.load_vcf(small_vcf_path, start=0, end=1000)
        s.iterative_start()
        assert s.arg is not None
        assert len(s.arg.sample_nodes) == 10

    def test_arg_has_recombinations(self, small_vcf_path):
        s = Sampler(Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4)
        s.set_seed(7)
        s.load_vcf(small_vcf_path, start=0, end=1000)
        s.iterative_start()
        # recombinations should have at least the two sentinels
        assert len(s.arg.recombinations) >= 2
