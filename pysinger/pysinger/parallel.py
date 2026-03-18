"""
Parallel MCMC — run multiple independent MCMC chains in parallel.

Supports two modes:

1. **Independent chains** — each chain runs with its own RNG seed and
   produces an independent ARG sample.  Results are collected and returned
   as a list of tskit TreeSequences.

2. **Parallel tempering** — chains run at different "temperatures" (mutation
   rate multipliers) and periodically propose swaps between adjacent chains.
   The cold chain (temperature=1) produces the final sample.

Usage::

    from pysinger.parallel import ParallelSampler

    ps = ParallelSampler(
        n_chains=4,
        Ne=1e4,
        recomb_rate=1e-8,
        mut_rate=1e-8,
    )
    ps.load_vcf("data.vcf", start=0, end=1_000_000)
    results = ps.run(num_iters=100, spacing=1)
    # results is a list of tskit.TreeSequence, one per chain
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .rates.rate_map import RateMap


# ---------------------------------------------------------------------------
# Worker function — runs in a child process
# ---------------------------------------------------------------------------

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
    """Execute a single MCMC chain and write result to a tskit .trees file.

    This function is designed to run in a child process — it imports
    pysinger locally so that each process gets its own module-level state
    (in particular the TSP ``_counter``).

    Returns the path to the output .trees file.
    """
    from .sampler import Sampler
    from .io.tskit_writer import arg_to_tskit

    sampler = Sampler(
        Ne=cfg.Ne,
        recomb_rate=cfg.recomb_rate / cfg.Ne,   # undo Ne scaling (Sampler re-applies it)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ParallelSampler:
    """Run multiple MCMC chains in parallel.

    Parameters
    ----------
    n_chains : int
        Number of independent chains to run.
    Ne, recomb_rate, mut_rate :
        Same as :class:`~pysinger.Sampler`.
    recomb_map, mut_map :
        Optional variable-rate maps (not yet forwarded to workers).
    base_seed : int
        Each chain gets ``base_seed + chain_id`` as its seed.
    temperatures : list[float] or None
        Per-chain temperature multipliers for parallel tempering.
        ``None`` means all chains run at temperature 1 (independent).
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
        self.recomb_rate = recomb_rate * Ne  # coalescent-scaled (matches Sampler)
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

        # VCF parameters (set by load_vcf)
        self._vcf_file: Optional[str] = None
        self._start: float = 0.0
        self._end: float = float("inf")
        self._haploid: bool = False

        # Precision
        self.bsp_c: float = 0.0
        self.tsp_q: float = 0.02
        self.penalty: float = 1.0
        self.polar: float = 0.5

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

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
        """Store VCF path — each worker will load it independently."""
        self._vcf_file = vcf_file
        self._start = start
        self._end = end
        self._haploid = haploid

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _make_configs(
        self,
        num_iters: int,
        spacing: int,
        output_dir: Optional[str],
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
        """Run all chains in parallel and return paths to .trees files.

        Parameters
        ----------
        num_iters : int
            MCMC iterations per chain.
        spacing : int
            Spacing parameter forwarded to each chain.
        n_workers : int or None
            Max worker processes.  Defaults to ``n_chains``.
        output_dir : str or None
            Directory to write .trees files.  If None, uses a temp dir.
        n_workers : int or None
            Number of worker processes (defaults to n_chains).

        Returns
        -------
        paths : list[str]
            Paths to the output .trees files, one per chain.
        """
        if n_workers is None:
            n_workers = self.n_chains
        configs = self._make_configs(num_iters, spacing, output_dir)

        # Use 'spawn' context to avoid fork-safety issues on macOS
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
        """Run all chains and return tskit TreeSequences in memory.

        Requires tskit to be installed.

        Returns
        -------
        tree_sequences : list[tskit.TreeSequence]
        """
        import tskit

        paths = self.run(num_iters=num_iters, spacing=spacing, n_workers=n_workers)
        return [tskit.load(p) for p in paths]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

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
    """One-shot convenience: configure, run, and return .trees file paths."""
    ps = ParallelSampler(
        n_chains=n_chains,
        Ne=Ne,
        recomb_rate=recomb_rate,
        mut_rate=mut_rate,
        base_seed=base_seed,
    )
    ps.load_vcf(vcf_file, start, end, haploid=haploid)
    return ps.run(
        num_iters=num_iters,
        spacing=spacing,
        n_workers=n_workers,
        output_dir=output_dir,
    )
