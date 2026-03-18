"""
Sampler — top-level MCMC sampler for pysinger.

Mirrors Sampler.cpp / Sampler.hpp.

Typical usage::

    sampler = Sampler(Ne=10000, recomb_rate=1e-8, mut_rate=1e-8)
    sampler.load_vcf("data/chr1.vcf", start=0, end=1_000_000)
    sampler.iterative_start()
    sampler.internal_sample(num_iters=100, spacing=1)
"""
from __future__ import annotations

import math
import random as stdlib_random
from typing import List, Optional, Set

import numpy as np

from .data.arg import ARG
from .data.node import Node
from .io.vcf_reader import read_vcf_phased, read_vcf_haploid
from .mcmc.threader import Threader
from .rates.rate_map import RateMap


class Sampler:
    """Bayesian ARG sampler.

    Parameters
    ----------
    Ne : float
        Haploid effective population size (used as a time-scaling factor).
    recomb_rate : float
        Per-base-pair per-generation recombination rate.  Scaled internally
        by Ne to get the coalescent-unit rate.
    mut_rate : float
        Per-base-pair per-generation mutation rate.  Scaled by Ne.
    recomb_map, mut_map : RateMap or None
        Variable-rate maps.  If provided, take precedence over the scalar
        rates for rho/theta computation.
    """

    def __init__(
        self,
        Ne: float = 1.0,
        recomb_rate: float = 0.0,
        mut_rate: float = 0.0,
        recomb_map: Optional[RateMap] = None,
        mut_map: Optional[RateMap] = None,
    ) -> None:
        self.Ne = Ne
        # Scale to coalescent units (× Ne)
        self.recomb_rate: float = recomb_rate * Ne
        self.mut_rate: float = mut_rate * Ne
        self.recomb_map: Optional[RateMap] = recomb_map
        self.mut_map: Optional[RateMap] = mut_map

        self.sequence_length: float = 0.0

        # BSP/TSP precision parameters
        self.bsp_c: float = 0.0
        self.tsp_q: float = 0.02

        # Emission model parameters (mirroring Threader_smc fields)
        self.penalty: float = 1.0
        self.polar: float = 0.5

        self.arg: Optional[ARG] = None
        self.sample_nodes: Set[Node] = set()
        # Ordered sequence for threading (shuffled after loading)
        self.ordered_sample_nodes: List[Node] = []

        self.sample_index: int = 0
        self.last_scale: float = 1.0
        self.last_arg_length: float = 0.0

        # Reproducible RNG
        self._rng = np.random.default_rng()
        self._seed: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_precision(self, c: float, q: float) -> None:
        self.bsp_c = c
        self.tsp_q = q

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Reset the global node counter so that results are reproducible
        # across multiple Sampler instances in the same Python session.
        import pysinger.hmm.tsp as _tsp_mod
        _tsp_mod._counter = 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_vcf(
        self,
        vcf_file: str,
        start: float = 0.0,
        end: float = float("inf"),
        haploid: bool = False,
    ) -> None:
        """Load genotype data from a VCF file.

        Parameters
        ----------
        vcf_file :
            Path to the .vcf file.
        start, end :
            Genomic region (half-open).
        haploid :
            If True, treat each column as one haplotype (no phased parsing).
        """
        if haploid:
            nodes, seq_len = read_vcf_haploid(vcf_file, start, end)
        else:
            nodes, seq_len = read_vcf_phased(vcf_file, start, end)

        self.sequence_length = seq_len
        self.sample_nodes = set(nodes)
        # Shuffle order deterministically
        shuffled = list(nodes)
        rng_state = np.random.default_rng(self._seed)
        rng_state.shuffle(shuffled)
        self.ordered_sample_nodes = shuffled

    # ------------------------------------------------------------------
    # ARG construction
    # ------------------------------------------------------------------

    def _build_singleton_arg(self) -> None:
        """Build an ARG containing only the first sample node.

        Mirrors Sampler::build_singleton_arg.
        """
        bin_size = max(1.0, 1.0 / self.recomb_rate) if self.recomb_rate > 0 else 100.0
        bin_size = min(bin_size, 100.0)
        first_node = self.ordered_sample_nodes[0]
        self.arg = ARG(self.Ne, self.sequence_length)
        self.arg.discretize(bin_size)
        self.arg.build_singleton_arg(first_node)
        if self.recomb_rate > 0 and self.mut_rate > 0:
            self.arg.compute_rhos_thetas(self.recomb_rate, self.mut_rate)
        elif self.recomb_map is not None and self.mut_map is not None:
            self.arg.compute_rhos_thetas(self.recomb_map, self.mut_map)

    def _make_threader(self) -> Threader:
        t = Threader(cutoff=self.bsp_c, gap=self.tsp_q)
        t.pe.penalty = self.penalty
        t.pe.ancestral_prob = self.polar
        t.set_rng(self._rng)
        return t

    # ------------------------------------------------------------------
    # Iterative initialisation
    # ------------------------------------------------------------------

    def iterative_start(self, max_retries: int = 5) -> None:
        """Thread all sample nodes one by one to build an initial ARG.

        Mirrors Sampler::iterative_start.

        If a threading step fails (e.g. HMM underflow on long sequences),
        the entire build is retried with a fresh RNG state up to
        *max_retries* times.
        """
        for attempt in range(max_retries):
            try:
                self._build_singleton_arg()
                for node in self.ordered_sample_nodes[1:]:
                    threader = self._make_threader()
                    threader.thread(self.arg, node)
                self._rescale()
                return  # success
            except RuntimeError:
                # Bump the RNG so the next attempt explores different paths
                self._rng = np.random.default_rng(self._seed + attempt + 1)
        raise RuntimeError(
            f"iterative_start failed after {max_retries} attempts "
            f"(sequence_length={self.sequence_length})"
        )

    # ------------------------------------------------------------------
    # MCMC sampling
    # ------------------------------------------------------------------

    def internal_sample(self, num_iters: int, spacing: int = 1) -> None:
        """Run *num_iters* MCMC iterations.

        Each iteration proposes at least ``spacing * sequence_length`` bp
        of re-threading moves.

        Mirrors Sampler::internal_sample.
        """
        while self.sample_index < num_iters:
            updated_length = 0.0
            while updated_length < spacing * self.arg.sequence_length:
                threader = self._make_threader()
                cut_point = self.arg.sample_internal_cut()
                try:
                    threader.internal_rethread(self.arg, cut_point)
                except Exception:
                    # If arg.remove() already ran (joining_branches is populated),
                    # restore the original lineage before clearing state.
                    if self.arg.joining_branches:
                        try:
                            self.arg.add(
                                self.arg.joining_branches,
                                self.arg.removed_branches,
                            )
                            self.arg.approx_sample_recombinations()
                        except Exception:
                            pass  # best effort; clear bookkeeping either way
                    self.arg.clear_remove_info()
                    break
                updated_length += (
                    self.arg.coordinates[threader.end_index]
                    - self.arg.coordinates[threader.start_index]
                )
                self.arg.clear_remove_info()
            self.last_arg_length = self.arg.get_arg_length()
            self.last_scale = self._rescale()
            self.sample_index += 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rescale(self) -> float:
        """Rescale ARG branch lengths to be consistent with mutation rate.

        Minimal version: compute a global scale factor from observed vs
        expected mutations and apply it to all node times.
        Mirrors Sampler::rescale (which calls Scaler::rescale).
        """
        if self.arg is None or self.mut_rate <= 0:
            return 1.0
        # Count total observed mutations from sample nodes.
        # (arg.node_set is not populated during threading; sample_nodes are.)
        # Use unique segregating site positions so each mutation event is
        # counted once (matching expected = mut_rate * branch_length).
        total_obs = len({
            pos
            for n in self.sample_nodes
            for pos in n.mutation_sites.keys()
            if pos >= 0
        })
        if total_obs == 0:
            return 1.0
        # Expected = mut_rate * total_branch_length
        total_branch = self.arg.get_arg_length()
        if total_branch <= 0:
            return 1.0
        # In SINGER's convention, 1 coalescent time unit = Ne generations
        # (haploid coalescent: rate = 1/Ne per gen → 1 unit = Ne gen).
        # self.mut_rate = mu * Ne = mutation probability per bp per time unit.
        expected = self.mut_rate * total_branch
        if expected <= 0:
            return 1.0
        scale = total_obs / expected
        # Collect all internal nodes by walking the ARG tree sequence.
        # Nodes that appear as parents in any tree are internal (non-leaf) nodes.
        sample_ids = {id(n) for n in self.sample_nodes}
        seen: set = set()
        internal_nodes = []
        tree = self.arg.get_tree_at(0.0)
        for n in tree.parents.values():
            if n.index >= 0 and id(n) not in seen and id(n) not in sample_ids:
                seen.add(id(n))
                internal_nodes.append(n)
        for pos, r in self.arg.recombinations.items():
            if 0 < pos < self.arg.sequence_length:
                tree.forward_update(r)
                for n in tree.parents.values():
                    if n.index >= 0 and id(n) not in seen and id(n) not in sample_ids:
                        seen.add(id(n))
                        internal_nodes.append(n)
        for n in internal_nodes:
            n.time *= scale
        # Also rescale recombination start_times, which are stored as floats
        # (not node references) and must stay consistent with node times.
        for pos, r in self.arg.recombinations.items():
            if 0 < pos < self.arg.sequence_length and r.start_time > 0:
                r.start_time *= scale
        return scale
