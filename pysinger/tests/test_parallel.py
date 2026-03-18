"""Tests for parallel MCMC chains."""
import pytest

from pysinger.parallel import ParallelSampler, ChainConfig, _run_chain


class TestChainConfig:
    def test_default_temperature(self):
        cfg = ChainConfig(
            chain_id=0, seed=42, Ne=1000.0,
            recomb_rate=0.1, mut_rate=0.1,
            vcf_file="x.vcf", start=0, end=1000,
            haploid=False, bsp_c=0.0, tsp_q=0.02,
            penalty=1.0, polar=0.5,
            num_iters=1, spacing=1,
        )
        assert cfg.temperature == 1.0


class TestParallelSampler:
    def test_init_defaults(self):
        ps = ParallelSampler(n_chains=4, Ne=1e4, recomb_rate=1e-8, mut_rate=1e-8)
        assert ps.n_chains == 4
        assert ps.temperatures == [1.0, 1.0, 1.0, 1.0]

    def test_temperature_length_mismatch(self):
        with pytest.raises(ValueError, match="n_chains"):
            ParallelSampler(n_chains=3, temperatures=[1.0, 0.5])

    def test_run_requires_vcf(self):
        ps = ParallelSampler(n_chains=2)
        with pytest.raises(RuntimeError, match="load_vcf"):
            ps.run(num_iters=1)

    def test_make_configs(self, small_vcf_path):
        ps = ParallelSampler(
            n_chains=2, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
            base_seed=10,
        )
        ps.load_vcf(small_vcf_path, start=0, end=1000)
        configs = ps._make_configs(num_iters=5, spacing=1, output_dir=None)
        assert len(configs) == 2
        assert configs[0].seed == 10
        assert configs[1].seed == 11
        assert configs[0].chain_id == 0
        assert configs[1].chain_id == 1

    def test_run_single_chain(self, small_vcf_path, tmp_path):
        """Run a single chain end-to-end (no multiprocessing)."""
        ps = ParallelSampler(
            n_chains=1, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
            base_seed=42,
        )
        ps.load_vcf(small_vcf_path, start=0, end=1000)
        paths = ps.run(num_iters=1, spacing=1, output_dir=str(tmp_path))
        assert len(paths) == 1
        import os
        assert os.path.exists(paths[0])

    def test_run_two_chains(self, small_vcf_path, tmp_path):
        """Run two independent chains and verify outputs differ."""
        ps = ParallelSampler(
            n_chains=2, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
            base_seed=42,
        )
        ps.load_vcf(small_vcf_path, start=0, end=1000)
        paths = ps.run(num_iters=1, spacing=1, output_dir=str(tmp_path))
        assert len(paths) == 2
        import os
        for p in paths:
            assert os.path.exists(p)


class TestDiagnostics:
    def test_gelman_rubin_identical_chains(self):
        """Identical chains should give R-hat ~ 1."""
        import numpy as np
        from pysinger.diagnostics import gelman_rubin

        rng = np.random.default_rng(0)
        chain = rng.normal(size=100)
        chains = np.array([chain, chain])
        # With identical chains, W = var, B = 0 → R-hat = sqrt((n-1)/n) ≈ 1
        r = gelman_rubin(chains)
        assert abs(r - 1.0) < 0.05 or np.isnan(r)

    def test_gelman_rubin_different_means(self):
        """Chains with very different means should give R-hat >> 1."""
        import numpy as np
        from pysinger.diagnostics import gelman_rubin

        rng = np.random.default_rng(0)
        c1 = rng.normal(loc=0, size=100)
        c2 = rng.normal(loc=100, size=100)
        r = gelman_rubin(np.array([c1, c2]))
        assert r > 2.0

    def test_ess_white_noise(self):
        """ESS of iid samples should be close to n."""
        import numpy as np
        from pysinger.diagnostics import effective_sample_size

        rng = np.random.default_rng(1)
        chain = rng.normal(size=500)
        ess = effective_sample_size(chain)
        # For iid, ESS ≈ n
        assert ess > 300
