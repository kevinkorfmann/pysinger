"""Tests for parallel MCMC: NRPT, temperature ladder, log-likelihood, diagnostics."""
import math
import pytest

import numpy as np

from pysinger.parallel import (
    ParallelSampler,
    NRPTSampler,
    ChainConfig,
    geometric_ladder,
    adaptive_ladder,
    log_likelihood,
)


# ======================================================================
# Temperature ladder
# ======================================================================

class TestGeometricLadder:
    def test_single_chain(self):
        assert geometric_ladder(1) == [1.0]

    def test_endpoints(self):
        betas = geometric_ladder(4, t_max=10.0)
        assert len(betas) == 4
        assert betas[-1] == 1.0  # cold chain
        assert betas[0] < betas[-1]  # hot chain has smaller beta

    def test_monotonically_increasing(self):
        betas = geometric_ladder(6, t_max=100.0)
        for i in range(len(betas) - 1):
            assert betas[i] < betas[i + 1]

    def test_two_chains(self):
        betas = geometric_ladder(2, t_max=5.0)
        assert betas[-1] == 1.0
        assert abs(betas[0] - 1.0 / 5.0) < 1e-10


class TestAdaptiveLadder:
    def test_no_change_at_target(self):
        """If all rates are at target, ladder should barely move."""
        betas = [0.1, 0.4, 0.7, 1.0]
        rates = [0.234, 0.234, 0.234]
        new = adaptive_ladder(betas, rates, target_rate=0.234)
        assert new[0] == betas[0]
        assert new[-1] == 1.0
        for i in range(len(new) - 1):
            assert new[i] < new[i + 1]

    def test_low_rates_spread_ladder(self):
        """Low acceptance rates should push interior betas apart."""
        betas = [0.1, 0.5, 0.9, 1.0]
        rates = [0.01, 0.01, 0.01]  # very low
        new = adaptive_ladder(betas, rates, target_rate=0.3, step_size=0.5)
        # Interior betas should decrease (spread out)
        assert new[1] < betas[1]

    def test_preserves_monotonicity(self):
        betas = [0.01, 0.3, 0.6, 1.0]
        rates = [0.9, 0.01, 0.9]  # mixed
        new = adaptive_ladder(betas, rates, target_rate=0.234)
        for i in range(len(new) - 1):
            assert new[i] < new[i + 1]


# ======================================================================
# Log-likelihood
# ======================================================================

class TestLogLikelihood:
    def test_none_arg(self):
        assert log_likelihood(None, 1.0) == 0.0

    def test_zero_mut_rate(self, small_vcf_path):
        from pysinger.sampler import Sampler
        s = Sampler(Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4)
        s.set_seed(42)
        s.load_vcf(small_vcf_path, start=0, end=1000)
        s.iterative_start()
        assert log_likelihood(s.arg, 0.0) == 0.0

    def test_finite_value(self, small_vcf_path):
        from pysinger.sampler import Sampler
        s = Sampler(Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4)
        s.set_seed(42)
        s.load_vcf(small_vcf_path, start=0, end=1000)
        s.iterative_start()
        ll = log_likelihood(s.arg, s.mut_rate)
        assert math.isfinite(ll)
        assert ll < 0  # log-likelihood should be negative


# ======================================================================
# NRPT Sampler
# ======================================================================

class TestNRPTSampler:
    def test_init_defaults(self):
        nrpt = NRPTSampler(n_chains=4)
        assert nrpt.n_chains == 4
        assert nrpt.betas[-1] == 1.0
        assert len(nrpt.betas) == 4

    def test_beta_length_mismatch(self):
        with pytest.raises(ValueError):
            NRPTSampler(n_chains=3, betas=[0.1, 1.0])

    def test_requires_vcf(self):
        nrpt = NRPTSampler(n_chains=2)
        with pytest.raises(RuntimeError, match="load_vcf"):
            nrpt.run(num_iters=1)

    def test_run_two_chains(self, small_vcf_path):
        """Run NRPT with 2 chains for 2 iterations."""
        nrpt = NRPTSampler(
            n_chains=2,
            Ne=1000.0,
            recomb_rate=1e-4,
            mut_rate=1e-4,
            base_seed=42,
            adapt_ladder=False,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        samplers = nrpt.run(num_iters=2, spacing=1)
        assert len(samplers) == 2
        assert all(s.arg is not None for s in samplers)

    def test_swap_history_recorded(self, small_vcf_path):
        """Verify swap attempts are recorded."""
        nrpt = NRPTSampler(
            n_chains=3,
            Ne=1000.0,
            recomb_rate=1e-4,
            mut_rate=1e-4,
            base_seed=7,
            adapt_ladder=False,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=4, spacing=1)
        assert len(nrpt.swap_history) == 4
        # Each entry has n_chains-1 = 2 elements
        assert all(len(h) == 2 for h in nrpt.swap_history)

    def test_deo_alternation(self, small_vcf_path):
        """DEO should alternate between even and odd pairs."""
        nrpt = NRPTSampler(
            n_chains=4,
            Ne=1000.0,
            recomb_rate=1e-4,
            mut_rate=1e-4,
            base_seed=42,
            adapt_ladder=False,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=4, spacing=1)
        # After 4 iterations, both even and odd pairs should have been attempted
        assert nrpt.swap_attempt_counts[0] > 0  # pair (0,1) - even
        assert nrpt.swap_attempt_counts[1] > 0  # pair (1,2) - odd

    def test_cold_chain_accessor(self, small_vcf_path):
        nrpt = NRPTSampler(
            n_chains=2, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=1)
        assert nrpt.cold_chain is nrpt.samplers[-1]

    def test_communication_barrier(self, small_vcf_path):
        nrpt = NRPTSampler(
            n_chains=3, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=4)
        barrier = nrpt.communication_barrier()
        assert math.isfinite(barrier)
        assert barrier >= 0

    def test_adaptive_ladder(self, small_vcf_path):
        """Run with adaptation and verify betas still valid."""
        nrpt = NRPTSampler(
            n_chains=3, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
            adapt_ladder=True,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=12, spacing=1, adapt_every=5)
        # Betas should still be monotonically increasing
        for i in range(len(nrpt.betas) - 1):
            assert nrpt.betas[i] < nrpt.betas[i + 1]
        assert nrpt.betas[-1] == 1.0

    def test_callback(self, small_vcf_path):
        """Verify callback is called each iteration."""
        calls = []
        def cb(it, sampler):
            calls.append(it)
        nrpt = NRPTSampler(
            n_chains=2, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=3, callback=cb)
        assert calls == [0, 1, 2]

    def test_log_likelihoods_tracked(self, small_vcf_path):
        nrpt = NRPTSampler(
            n_chains=2, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
        )
        nrpt.load_vcf(small_vcf_path, start=0, end=1000)
        nrpt.run(num_iters=3)
        assert len(nrpt.log_likelihoods) == 3
        assert all(len(lls) == 2 for lls in nrpt.log_likelihoods)
        assert all(
            math.isfinite(ll)
            for lls in nrpt.log_likelihoods
            for ll in lls
        )


# ======================================================================
# ParallelSampler (independent chains via multiprocessing)
# ======================================================================

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

    def test_run_single_chain(self, small_vcf_path, tmp_path):
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
        ps = ParallelSampler(
            n_chains=2, Ne=1000.0, recomb_rate=1e-4, mut_rate=1e-4,
            base_seed=42,
        )
        ps.load_vcf(small_vcf_path, start=0, end=1000)
        paths = ps.run(num_iters=1, spacing=1, output_dir=str(tmp_path))
        assert len(paths) == 2


# ======================================================================
# Diagnostics
# ======================================================================

class TestDiagnostics:
    def test_gelman_rubin_identical_chains(self):
        from pysinger.diagnostics import gelman_rubin
        rng = np.random.default_rng(0)
        chain = rng.normal(size=100)
        chains = np.array([chain, chain])
        r = gelman_rubin(chains)
        assert abs(r - 1.0) < 0.05 or np.isnan(r)

    def test_gelman_rubin_different_means(self):
        from pysinger.diagnostics import gelman_rubin
        rng = np.random.default_rng(0)
        c1 = rng.normal(loc=0, size=100)
        c2 = rng.normal(loc=100, size=100)
        r = gelman_rubin(np.array([c1, c2]))
        assert r > 2.0

    def test_ess_white_noise(self):
        from pysinger.diagnostics import effective_sample_size
        rng = np.random.default_rng(1)
        chain = rng.normal(size=500)
        ess = effective_sample_size(chain)
        assert ess > 300
