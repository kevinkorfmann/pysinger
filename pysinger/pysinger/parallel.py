"""
Parallel MCMC for Bayesian ARG sampling.

Implements three mathematically grounded parallel MCMC strategies:

1. **Non-Reversible Parallel Tempering (NRPT)** — Syed et al. (JRSS-B 2022).
   Chains at inverse temperatures beta_0 < ... < beta_{N-1} = 1 target tempered
   posteriors pi_beta(ARG) proportional to L(data|ARG)^beta * P(ARG).  A
   deterministic even-odd (DEO) swap schedule replaces the standard random-scan,
   yielding ballistic rather than diffusive movement through temperature space.
   The temperature ladder is adapted online to equalise swap acceptance rates
   (minimising the global communication barrier Lambda).

2. **Independent chains** — embarrassingly parallel sampling for diagnostics.
   Each chain has its own seed; results are combined for R-hat / ESS.

3. **Multi-particle BSP (SMC-within-MCMC foundation)** — replaces the single
   forward-backward path in BSP with K particles, improving proposal quality
   at the cost of K-fold computation per threading step.

Mathematical background
-----------------------
SINGER's MH acceptance ratio is ``old_height / new_height`` — the coalescent
prior ratio P(ARG')/P(ARG).  The likelihood L(data|ARG) is absorbed into the
BSP/TSP proposal via emission probabilities that depend on theta.

For tempering at inverse temperature beta:
  - Target:   pi_beta(ARG) propto L(data|ARG)^beta * P(ARG)
  - Proposal: BSP/TSP with theta_beta = beta * theta  (tempers the likelihood
    component of the proposal)
  - Because q_beta ~ L^beta, the MH ratio stays ~ P(new)/P(old), i.e. the
    acceptance ratio is the same as the cold chain.

For swap moves between chains i, j with states x_i, x_j:
  alpha_swap = min(1, exp((beta_i - beta_j) * (log_L(x_j) - log_L(x_i))))

References
----------
- Syed, Bouchard-Cote, Deligiannidis & Doucet, "Non-Reversible Parallel
  Tempering," JRSS-B 84(2):321-350, 2022.
- Altekar et al., "Parallel Metropolis coupled MCMC for Bayesian phylogenetic
  inference," Bioinformatics 20(3):407-415, 2004.
- Del Moral, Doucet & Jasra, "Sequential Monte Carlo samplers," JRSS-B
  68(3):411-436, 2006.
"""
from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .sampler import Sampler
from .rates.rate_map import RateMap

if TYPE_CHECKING:
    from .data.arg import ARG


# ======================================================================
# Log-likelihood computation
# ======================================================================

def log_likelihood(arg: "ARG", mut_rate: float) -> float:
    """Approximate log P(data | ARG) under the infinite-sites Poisson model.

    For each genomic bin of width w with s segregating sites and total
    branch length L:

        log P(s | ARG) = -theta*L + s*log(theta*L) - log(s!)

    where theta = mut_rate (already in coalescent units).

    We sum over all bins.  The s! term is constant w.r.t. the ARG and
    drops out of likelihood ratios, so we omit it.
    """
    if arg is None or mut_rate <= 0:
        return 0.0

    total_ll = 0.0
    n_bins = len(arg.coordinates) - 1

    tree = arg.get_tree_at(0.0)

    recomb_keys = sorted(
        k for k in arg.recombinations.keys()
        if 0 < k < arg.sequence_length
    )
    recomb_idx = 0

    for i in range(n_bins):
        pos = arg.coordinates[i]
        next_pos = arg.coordinates[i + 1]
        w = next_pos - pos

        # Advance tree through recombinations up to this position
        while recomb_idx < len(recomb_keys) and recomb_keys[recomb_idx] <= pos:
            tree.forward_update(arg.recombinations[recomb_keys[recomb_idx]])
            recomb_idx += 1

        branch_length = tree.length()
        theta_L = mut_rate * branch_length * w

        # Count segregating sites in [pos, next_pos)
        n_muts = 0
        if hasattr(arg, 'mutation_sites'):
            for m in arg.mutation_sites:
                if pos <= m < next_pos:
                    n_muts += 1

        if theta_L > 0:
            total_ll += -theta_L + n_muts * math.log(theta_L)
        elif n_muts > 0:
            total_ll += -1e10  # impossible configuration

    return total_ll


# ======================================================================
# Temperature ladder utilities
# ======================================================================

def geometric_ladder(n_chains: int, t_max: float = 10.0) -> List[float]:
    """Geometric temperature ladder: T_i = t_max^{i/(N-1)}, beta_i = 1/T_i.

    Returns inverse temperatures (betas) in ascending order, with
    beta[-1] = 1.0 (cold chain).
    """
    if n_chains == 1:
        return [1.0]
    betas = []
    for i in range(n_chains):
        T = t_max ** ((n_chains - 1 - i) / (n_chains - 1))
        betas.append(1.0 / T)
    return betas


def adaptive_ladder(
    betas: List[float],
    swap_accept_rates: List[float],
    target_rate: float = 0.234,
    step_size: float = 0.1,
) -> List[float]:
    """Adapt the temperature ladder to equalise swap acceptance rates.

    Adjusts interior betas so that each adjacent pair has acceptance
    rate close to ``target_rate``.  The cold (beta=1) and hot (beta_0)
    endpoints are fixed.

    The optimal target for Gaussian-like posteriors is ~0.234 (Roberts
    et al.), though for multimodal targets higher rates (~0.3-0.5) may
    be preferable.

    Parameters
    ----------
    betas : list of float
        Current inverse temperatures, ascending.
    swap_accept_rates : list of float
        Observed acceptance rates for each adjacent pair (length N-1).
    target_rate : float
        Target swap acceptance rate.
    step_size : float
        Learning rate for adaptation (Robbins-Monro style).

    Returns
    -------
    new_betas : list of float
        Adapted ladder.
    """
    n = len(betas)
    if n <= 2:
        return list(betas)

    new_betas = list(betas)
    # Work in log-beta space for stability
    log_betas = [math.log(max(b, 1e-10)) for b in betas]

    for k in range(1, n - 1):
        if k - 1 < len(swap_accept_rates):
            delta = swap_accept_rates[k - 1] - target_rate
            # Robbins-Monro: move log_beta[k] to equalise acceptance
            log_betas[k] += step_size * delta

    # Ensure monotonicity
    for k in range(1, n - 1):
        log_betas[k] = max(log_betas[k], log_betas[k - 1] + 0.01)
        log_betas[k] = min(log_betas[k], log_betas[k + 1] - 0.01)

    new_betas = [math.exp(lb) for lb in log_betas]
    new_betas[0] = betas[0]
    new_betas[-1] = 1.0
    return new_betas


# ======================================================================
# Non-Reversible Parallel Tempering (NRPT)
# ======================================================================

class NRPTSampler:
    """Non-Reversible Parallel Tempering with DEO swap schedule.

    All chains run in a single process to enable swap moves that require
    access to both chains' ARG states.  For CPU parallelism of the
    within-chain HMM passes, see ``ParallelSampler``.

    Parameters
    ----------
    n_chains : int
        Number of tempered chains.
    Ne, recomb_rate, mut_rate :
        Same as :class:`~pysinger.Sampler`.
    betas : list[float] or None
        Inverse temperature ladder.  If None, a geometric ladder is used.
    base_seed : int
        Chain i gets seed ``base_seed + i``.
    adapt_ladder : bool
        Whether to adapt the temperature ladder online.
    swap_interval : int
        Number of within-chain MCMC sub-iterations between swap attempts.
    """

    def __init__(
        self,
        n_chains: int = 4,
        Ne: float = 1.0,
        recomb_rate: float = 0.0,
        mut_rate: float = 0.0,
        betas: Optional[List[float]] = None,
        base_seed: int = 42,
        adapt_ladder: bool = True,
        swap_interval: int = 1,
    ) -> None:
        self.n_chains = n_chains
        self.Ne = Ne
        self.recomb_rate = recomb_rate
        self.mut_rate = mut_rate
        self.base_seed = base_seed
        self.adapt_ladder = adapt_ladder
        self.swap_interval = swap_interval

        if betas is not None:
            if len(betas) != n_chains:
                raise ValueError(
                    f"len(betas)={len(betas)} != n_chains={n_chains}"
                )
            self.betas = list(betas)
        else:
            self.betas = geometric_ladder(n_chains)

        # Precision params
        self.bsp_c: float = 0.0
        self.tsp_q: float = 0.02
        self.penalty: float = 1.0
        self.polar: float = 0.5

        # VCF params (set by load_vcf)
        self._vcf_file: Optional[str] = None
        self._start: float = 0.0
        self._end: float = float("inf")
        self._haploid: bool = False

        # Runtime state (populated by run)
        self.samplers: List[Sampler] = []
        self.swap_history: List[List[int]] = []  # per-iteration swap outcomes
        self.swap_accept_counts: Optional[np.ndarray] = None
        self.swap_attempt_counts: Optional[np.ndarray] = None
        self.log_likelihoods: List[List[float]] = []  # per-iter, per-chain

        # DEO direction state: +1 = even pairs first, -1 = odd pairs first
        self._deo_direction: int = 1

        self._rng = np.random.default_rng(base_seed + n_chains + 1000)

    def set_precision(self, c: float, q: float) -> None:
        self.bsp_c = c
        self.tsp_q = q

    def load_vcf(
        self,
        vcf_file: str,
        start: float = 0.0,
        end: float = float("inf"),
        haploid: bool = False,
    ) -> None:
        self._vcf_file = vcf_file
        self._start = start
        self._end = end
        self._haploid = haploid

    # ------------------------------------------------------------------
    # Chain initialisation
    # ------------------------------------------------------------------

    def _init_chains(self) -> None:
        """Create and initialise one Sampler per temperature."""
        if self._vcf_file is None:
            raise RuntimeError("Call load_vcf() before run()")

        self.samplers = []
        for i in range(self.n_chains):
            beta = self.betas[i]
            s = Sampler(
                Ne=self.Ne,
                recomb_rate=self.recomb_rate / self.Ne,
                # Temper: scale mut_rate by beta so BSP/TSP emissions
                # target L^beta rather than L.
                mut_rate=self.mut_rate / self.Ne * beta,
            )
            s.set_precision(self.bsp_c, self.tsp_q)
            s.penalty = self.penalty
            s.polar = self.polar
            s.set_seed(self.base_seed + i)
            s.load_vcf(self._vcf_file, self._start, self._end, self._haploid)
            s.iterative_start()
            self.samplers.append(s)

        self.swap_accept_counts = np.zeros(self.n_chains - 1)
        self.swap_attempt_counts = np.zeros(self.n_chains - 1)

    # ------------------------------------------------------------------
    # DEO swap schedule
    # ------------------------------------------------------------------

    def _propose_swaps(self) -> List[int]:
        """Propose swaps using the DEO (deterministic even-odd) schedule.

        In the DEO scheme (Syed et al. 2022), even iterations swap pairs
        (0,1), (2,3), (4,5), ... and odd iterations swap pairs (1,2),
        (3,4), (5,6), ...  This is non-reversible: the parity alternates
        deterministically rather than being chosen at random.

        Returns list of swap outcomes: 1 = accepted, 0 = rejected,
        for each adjacent pair.
        """
        outcomes = [0] * (self.n_chains - 1)

        # Determine which pairs to attempt this round
        if self._deo_direction == 1:
            pairs = list(range(0, self.n_chains - 1, 2))  # even pairs
        else:
            pairs = list(range(1, self.n_chains - 1, 2))  # odd pairs

        for k in pairs:
            self.swap_attempt_counts[k] += 1

            beta_i = self.betas[k]
            beta_j = self.betas[k + 1]

            # Compute log-likelihoods at the COLD chain's mut_rate
            # (the tempered mut_rate is beta*mut_rate, so we use the
            # untempered rate for the swap criterion).
            ll_i = log_likelihood(self.samplers[k].arg, self.mut_rate)
            ll_j = log_likelihood(self.samplers[k + 1].arg, self.mut_rate)

            # Swap acceptance:
            # alpha = min(1, exp((beta_i - beta_j) * (log_L(x_j) - log_L(x_i))))
            log_alpha = (beta_i - beta_j) * (ll_j - ll_i)
            log_u = math.log(max(self._rng.uniform(), 1e-300))

            if log_u < log_alpha:
                # Accept: swap the ARG states (and associated sampler state)
                self.samplers[k], self.samplers[k + 1] = (
                    self.samplers[k + 1],
                    self.samplers[k],
                )
                # Update each swapped sampler's mut_rate to match its new
                # temperature position.
                self.samplers[k].mut_rate = self.mut_rate * self.betas[k]
                self.samplers[k + 1].mut_rate = self.mut_rate * self.betas[k + 1]

                self.swap_accept_counts[k] += 1
                outcomes[k] = 1

        # Flip DEO direction
        self._deo_direction *= -1
        return outcomes

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        num_iters: int = 10,
        spacing: int = 1,
        adapt_every: int = 10,
        target_swap_rate: float = 0.234,
        callback: Optional[Callable[[int, "NRPTSampler"], None]] = None,
    ) -> List[Sampler]:
        """Run NRPT for ``num_iters`` macro-iterations.

        Each macro-iteration consists of:
          1. ``swap_interval`` within-chain MCMC sub-iterations per chain
          2. One round of DEO swap proposals
          3. (Optional) temperature ladder adaptation

        Parameters
        ----------
        num_iters : int
            Number of macro-iterations.
        spacing : int
            Forwarded to each chain's ``internal_sample``.
        adapt_every : int
            Adapt the temperature ladder every this many iterations.
        target_swap_rate : float
            Target swap acceptance rate for ladder adaptation.
        callback : callable or None
            Called as ``callback(iteration, self)`` after each iteration.

        Returns
        -------
        samplers : list[Sampler]
            The sampler objects, ordered by temperature (last = cold chain).
        """
        self._init_chains()

        for iteration in range(num_iters):
            # 1. Within-chain sampling
            for chain in self.samplers:
                chain.internal_sample(
                    num_iters=chain.sample_index + self.swap_interval,
                    spacing=spacing,
                )

            # 2. DEO swap proposals
            if self.n_chains > 1:
                outcomes = self._propose_swaps()
                self.swap_history.append(outcomes)

            # 3. Record log-likelihoods (at untempered rate)
            lls = [
                log_likelihood(s.arg, self.mut_rate) for s in self.samplers
            ]
            self.log_likelihoods.append(lls)

            # 4. Adaptive ladder
            if (
                self.adapt_ladder
                and self.n_chains > 2
                and (iteration + 1) % adapt_every == 0
            ):
                rates = []
                for k in range(self.n_chains - 1):
                    if self.swap_attempt_counts[k] > 0:
                        rates.append(
                            self.swap_accept_counts[k]
                            / self.swap_attempt_counts[k]
                        )
                    else:
                        rates.append(0.5)

                new_betas = adaptive_ladder(
                    self.betas,
                    rates,
                    target_rate=target_swap_rate,
                    step_size=max(0.01, 1.0 / (iteration + 1)),
                )
                # Update betas and re-scale each chain's mut_rate
                for k in range(self.n_chains):
                    self.betas[k] = new_betas[k]
                    self.samplers[k].mut_rate = self.mut_rate * new_betas[k]

            if callback is not None:
                callback(iteration, self)

        return self.samplers

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def cold_chain(self) -> Sampler:
        """The chain at beta=1 (the target posterior)."""
        return self.samplers[-1]

    @property
    def swap_acceptance_rates(self) -> List[float]:
        """Observed swap acceptance rates per adjacent pair."""
        if self.swap_attempt_counts is None:
            return []
        rates = []
        for k in range(self.n_chains - 1):
            if self.swap_attempt_counts[k] > 0:
                rates.append(
                    float(self.swap_accept_counts[k] / self.swap_attempt_counts[k])
                )
            else:
                rates.append(0.0)
        return rates

    def communication_barrier(self) -> float:
        """Estimate the global communication barrier Lambda.

        Lambda = sum_{k=0}^{N-2} (1 - acceptance_rate_k)

        This quantity measures the expected number of swap rejections a
        sample must pass through to travel from the hottest to the
        coldest chain.  Lower is better.  Syed et al. show that NRPT's
        round-trip rate scales as 1/Lambda vs 1/Lambda^2 for reversible PT.
        """
        rates = self.swap_acceptance_rates
        if not rates:
            return float("inf")
        return sum(1.0 - r for r in rates)


# ======================================================================
# Independent chains via multiprocessing (kept for backward compat)
# ======================================================================

@dataclass
class ChainConfig:
    """Serialisable configuration for a single MCMC chain."""

    chain_id: int
    seed: int
    Ne: float
    recomb_rate: float
    mut_rate: float
    vcf_file: str
    start: float
    end: float
    haploid: bool
    bsp_c: float
    tsp_q: float
    penalty: float
    polar: float
    num_iters: int
    spacing: int
    temperature: float = 1.0
    output_path: Optional[str] = None


def _run_chain(cfg: ChainConfig) -> str:
    """Execute a single MCMC chain in a child process."""
    from .sampler import Sampler
    from .io.tskit_writer import arg_to_tskit

    sampler = Sampler(
        Ne=cfg.Ne,
        recomb_rate=cfg.recomb_rate / cfg.Ne,
        mut_rate=cfg.mut_rate / cfg.Ne * cfg.temperature,
    )
    sampler.set_precision(cfg.bsp_c, cfg.tsp_q)
    sampler.penalty = cfg.penalty
    sampler.polar = cfg.polar
    sampler.set_seed(cfg.seed)
    sampler.load_vcf(cfg.vcf_file, cfg.start, cfg.end, cfg.haploid)

    sampler.iterative_start()
    sampler.internal_sample(num_iters=cfg.num_iters, spacing=cfg.spacing)

    ts = arg_to_tskit(sampler.arg, Ne=cfg.Ne)
    out = cfg.output_path or os.path.join(
        tempfile.gettempdir(),
        f"pysinger_chain_{cfg.chain_id}_{os.getpid()}.trees",
    )
    ts.dump(out)
    return out


class ParallelSampler:
    """Run multiple independent MCMC chains via multiprocessing.

    For proper parallel tempering with swap moves, use ``NRPTSampler``
    instead.  This class runs chains that never communicate — useful
    for convergence diagnostics (R-hat, ESS) but not for improving
    mixing.
    """

    def __init__(
        self,
        n_chains: int = 4,
        Ne: float = 1.0,
        recomb_rate: float = 0.0,
        mut_rate: float = 0.0,
        recomb_map: Optional[RateMap] = None,
        mut_map: Optional[RateMap] = None,
        base_seed: int = 42,
        temperatures: Optional[List[float]] = None,
    ) -> None:
        self.n_chains = n_chains
        self.Ne = Ne
        self.recomb_rate = recomb_rate * Ne
        self.mut_rate = mut_rate * Ne
        self.recomb_map = recomb_map
        self.mut_map = mut_map
        self.base_seed = base_seed

        if temperatures is not None:
            if len(temperatures) != n_chains:
                raise ValueError(
                    f"len(temperatures)={len(temperatures)} != n_chains={n_chains}"
                )
            self.temperatures = temperatures
        else:
            self.temperatures = [1.0] * n_chains

        self._vcf_file: Optional[str] = None
        self._start: float = 0.0
        self._end: float = float("inf")
        self._haploid: bool = False
        self.bsp_c: float = 0.0
        self.tsp_q: float = 0.02
        self.penalty: float = 1.0
        self.polar: float = 0.5

    def set_precision(self, c: float, q: float) -> None:
        self.bsp_c = c
        self.tsp_q = q

    def load_vcf(
        self,
        vcf_file: str,
        start: float = 0.0,
        end: float = float("inf"),
        haploid: bool = False,
    ) -> None:
        self._vcf_file = vcf_file
        self._start = start
        self._end = end
        self._haploid = haploid

    def _make_configs(
        self, num_iters: int, spacing: int, output_dir: Optional[str],
    ) -> List[ChainConfig]:
        if self._vcf_file is None:
            raise RuntimeError("Call load_vcf() before run()")
        configs = []
        for i in range(self.n_chains):
            out = None
            if output_dir is not None:
                out = os.path.join(output_dir, f"chain_{i}.trees")
            configs.append(
                ChainConfig(
                    chain_id=i,
                    seed=self.base_seed + i,
                    Ne=self.Ne,
                    recomb_rate=self.recomb_rate,
                    mut_rate=self.mut_rate,
                    vcf_file=self._vcf_file,
                    start=self._start,
                    end=self._end,
                    haploid=self._haploid,
                    bsp_c=self.bsp_c,
                    tsp_q=self.tsp_q,
                    penalty=self.penalty,
                    polar=self.polar,
                    num_iters=num_iters,
                    spacing=spacing,
                    temperature=self.temperatures[i],
                    output_path=out,
                )
            )
        return configs

    def run(
        self,
        num_iters: int = 10,
        spacing: int = 1,
        n_workers: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Run all chains and return paths to .trees files."""
        if n_workers is None:
            n_workers = self.n_chains
        configs = self._make_configs(num_iters, spacing, output_dir)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            paths = pool.map(_run_chain, configs)
        return paths

    def run_and_load(
        self,
        num_iters: int = 10,
        spacing: int = 1,
        n_workers: Optional[int] = None,
    ):
        """Run all chains and return tskit TreeSequences."""
        import tskit
        paths = self.run(num_iters=num_iters, spacing=spacing, n_workers=n_workers)
        return [tskit.load(p) for p in paths]


# ======================================================================
# Convenience
# ======================================================================

def run_parallel_chains(
    vcf_file: str,
    start: float,
    end: float,
    Ne: float = 1e4,
    recomb_rate: float = 1e-8,
    mut_rate: float = 1e-8,
    n_chains: int = 4,
    num_iters: int = 10,
    spacing: int = 1,
    base_seed: int = 42,
    haploid: bool = False,
    output_dir: Optional[str] = None,
    n_workers: Optional[int] = None,
) -> List[str]:
    """One-shot convenience: configure, run independent chains, return paths."""
    ps = ParallelSampler(
        n_chains=n_chains, Ne=Ne, recomb_rate=recomb_rate,
        mut_rate=mut_rate, base_seed=base_seed,
    )
    ps.load_vcf(vcf_file, start, end, haploid=haploid)
    return ps.run(
        num_iters=num_iters, spacing=spacing,
        n_workers=n_workers, output_dir=output_dir,
    )
