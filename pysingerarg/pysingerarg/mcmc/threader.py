"""
Threader — MCMC move: remove a lineage and re-thread it.

Mirrors Threader_smc.cpp / Threader_smc.hpp.

Two public entry points:
  thread(arg, node)            — add a new leaf node (initial threading)
  internal_rethread(arg, cut)  — MCMC move with Metropolis acceptance
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from sortedcontainers import SortedDict

from ..data.branch import Branch
from ..data.node import Node
from ..hmm.bsp import BSP
from ..hmm.tsp import TSP
from ..hmm.emission import BinaryEmission, PolarEmission

if TYPE_CHECKING:
    from ..data.arg import ARG
    from ..data.recombination import Recombination


class Threader:
    """BSP + TSP threader for adding/rethreading a lineage in an ARG.

    Parameters
    ----------
    cutoff : float
        BSP state-space pruning threshold (bsp_c in C++).
    gap : float
        TSP time grid quantile gap (tsp_q in C++).
    """

    def __init__(self, cutoff: float = 0.0, gap: float = 0.02) -> None:
        self.cutoff = cutoff
        self.gap = gap

        self.bsp: BSP = BSP()
        self.tsp: TSP = TSP()

        # Emission models: polar (BSP) and binary (TSP)
        self.pe: PolarEmission = PolarEmission()
        self.be: BinaryEmission = BinaryEmission()

        self.cut_time: float = 0.0
        self.start: float = 0.0
        self.end: float = 0.0
        self.start_index: int = 0
        self.end_index: int = 0

        # Results of a threading run
        self.new_joining_branches: SortedDict = SortedDict()
        self.added_branches: SortedDict = SortedDict()

        self._rng: np.random.Generator = np.random.default_rng()

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.bsp.set_rng(rng)
        self.tsp.set_rng(rng)

    def _random(self) -> float:
        return float(self._rng.uniform())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def thread(self, arg: "ARG", node: Node) -> None:
        """Add *node* as a new leaf and thread its lineage through *arg*.

        Mirrors Threader_smc::thread.
        """
        self.cut_time = 0.0
        arg.cut_time = 0.0
        arg.add_sample(node)
        self._get_boundary(arg)
        self._run_bsp(arg)
        self._sample_joining_branches(arg)
        self._run_tsp(arg)
        self._sample_joining_points(arg)
        arg.add(self.new_joining_branches, self.added_branches)
        arg.approx_sample_recombinations()
        arg.clear_remove_info()

    def internal_rethread(
        self,
        arg: "ARG",
        cut_point: Tuple[float, Branch, float],
    ) -> None:
        """MCMC move: remove and re-thread a lineage segment.

        Proposes a new threading, accepts with Metropolis ratio.
        Mirrors Threader_smc::internal_rethread.
        """
        self.cut_time = cut_point[2]
        arg.remove(cut_point)
        self._get_boundary(arg)
        self._set_check_points(arg)
        self._run_bsp(arg)
        self._sample_joining_branches(arg)
        self._run_tsp(arg)
        self._sample_joining_points(arg)
        ar = self._acceptance_ratio(arg)
        if self._random() < ar:
            arg.add(self.new_joining_branches, self.added_branches)
        else:
            arg.add(arg.joining_branches, arg.removed_branches)
        arg.approx_sample_recombinations()
        arg.clear_remove_info()

    # ------------------------------------------------------------------
    # Private: boundary and check points
    # ------------------------------------------------------------------

    def _get_boundary(self, arg: "ARG") -> None:
        self.start = arg.start
        self.end = arg.end
        self.start_index = arg.get_index(self.start)
        self.end_index = arg.get_index(self.end)

    def _set_check_points(self, arg: "ARG") -> None:
        check_points = arg.get_check_points()
        self.bsp.set_check_points(check_points)
        self.tsp.set_check_points(check_points)

    # ------------------------------------------------------------------
    # Private: BSP forward pass
    # ------------------------------------------------------------------

    def _run_bsp(self, arg: "ARG") -> None:
        """Run the BSP forward pass over *arg*.

        Mirrors Threader_smc::run_BSP.
        """
        self.bsp.reserve_memory(self.end_index - self.start_index)
        self.bsp.set_cutoff(self.cutoff)
        self.bsp.set_emission(self.pe)
        self.bsp.start(arg.start_tree, self.cut_time)

        recomb_it = arg.recombinations.irange(
            None, None, inclusive=(False, True)
        )
        # Advance past start
        recomb_keys = list(arg.recombinations.irange(
            self.start, None, inclusive=(False, True)
        ))
        recomb_idx = 0

        mut_keys = sorted(m for m in arg.mutation_sites if m >= self.start)
        mut_idx = 0

        removed_items = list(arg.removed_branches.items())
        query_idx = 0

        query_node: Optional[Node] = None

        for i in range(self.start_index, self.end_index):
            pos = arg.coordinates[i]

            # Advance query node
            if query_idx < len(removed_items) and pos == removed_items[query_idx][0]:
                query_node = removed_items[query_idx][1].lower_node
                query_idx += 1

            # Recombination or forward step
            if recomb_idx < len(recomb_keys) and pos == recomb_keys[recomb_idx]:
                r = arg.recombinations[recomb_keys[recomb_idx]]
                recomb_idx += 1
                self.bsp.transfer(r)
            elif pos != self.start:
                self.bsp.forward(arg.rhos[i - 1])

            # Collect mutations in [pos, next_pos)
            next_pos = arg.coordinates[i + 1]
            mut_set: Set[float] = set()
            while mut_idx < len(mut_keys) and mut_keys[mut_idx] < next_pos:
                mut_set.add(mut_keys[mut_idx])
                mut_idx += 1

            if mut_set:
                self.bsp.mut_emit(
                    arg.thetas[i],
                    next_pos - pos,
                    mut_set,
                    query_node,
                )
            else:
                self.bsp.null_emit(arg.thetas[i], query_node)

        # Sanity check at end boundary
        if self.end in self.bsp.check_points:
            r = arg.recombinations[self.end]
            self.bsp.sanity_check(r)

    # ------------------------------------------------------------------
    # Private: TSP forward pass
    # ------------------------------------------------------------------

    def _run_tsp(self, arg: "ARG") -> None:
        """Run the TSP forward pass over the sampled joining branches.

        Mirrors Threader_smc::run_TSP.
        """
        self.tsp.reserve_memory(self.end_index - self.start_index)
        self.tsp.set_gap(self.gap)
        self.tsp.set_emission(self.be)

        start_branch = self.new_joining_branches.peekitem(0)[1]  # first value
        self.tsp.start(start_branch, self.cut_time)

        recomb_keys = list(arg.recombinations.irange(
            self.start, None, inclusive=(False, True)
        ))
        recomb_idx = 0

        join_keys = list(self.new_joining_branches.irange(
            self.start, None, inclusive=(False, True)
        ))
        join_idx = 0

        mut_keys = sorted(m for m in arg.mutation_sites if m >= self.start)
        mut_idx = 0

        removed_items = list(arg.removed_branches.items())
        query_idx = 0

        query_node: Optional[Node] = None
        prev_branch = start_branch
        next_branch = start_branch

        for i in range(self.start_index, self.end_index):
            pos = arg.coordinates[i]

            # Advance query node
            if query_idx < len(removed_items) and pos == removed_items[query_idx][0]:
                query_node = removed_items[query_idx][1].lower_node
                query_idx += 1

            # Advance joining branch
            if join_idx < len(join_keys) and pos == join_keys[join_idx]:
                next_branch = self.new_joining_branches[join_keys[join_idx]]
                join_idx += 1

            # Transfer / recombine / forward
            if recomb_idx < len(recomb_keys) and pos == recomb_keys[recomb_idx]:
                r = arg.recombinations[recomb_keys[recomb_idx]]
                recomb_idx += 1
                self.tsp.transfer(r, prev_branch, next_branch)
                prev_branch = next_branch
            elif prev_branch is not next_branch:
                self.tsp.recombine(prev_branch, next_branch)
                prev_branch = next_branch
            elif pos != self.start:
                self.tsp.forward(arg.rhos[i])

            # Collect mutations in [pos, next_pos)
            next_pos = arg.coordinates[i + 1]
            mut_set: Set[float] = set()
            while mut_idx < len(mut_keys) and mut_keys[mut_idx] < next_pos:
                mut_set.add(mut_keys[mut_idx])
                mut_idx += 1

            if mut_set:
                self.tsp.mut_emit(
                    arg.thetas[i],
                    next_pos - pos,
                    mut_set,
                    query_node,
                )
            else:
                self.tsp.null_emit(arg.thetas[i], query_node)

        # Sanity check at end boundary
        if self.end in self.tsp.check_points:
            r = arg.recombinations[self.end]
            self.tsp._sanity_check(r)

    # ------------------------------------------------------------------
    # Private: sampling
    # ------------------------------------------------------------------

    def _sample_joining_branches(self, arg: "ARG") -> None:
        """Sample joining branches from BSP traceback.

        Mirrors Threader_smc::sample_joining_branches.
        """
        self.new_joining_branches = self.bsp.sample_joining_branches(
            self.start_index, arg.coordinates
        )

    def _sample_joining_points(self, arg: "ARG") -> None:
        """Sample joining nodes from TSP traceback and build added_branches.

        Mirrors Threader_smc::sample_joining_points.
        """
        added_nodes: Dict[float, Optional[Node]] = self.tsp.sample_joining_nodes(
            self.start_index, arg.coordinates
        )
        self.added_branches = SortedDict()
        for x, added_node in added_nodes.items():
            if added_node is None:
                self.added_branches[x] = Branch()  # null sentinel at sequence_length boundary
            else:
                query_node = arg.get_query_node_at(x)
                self.added_branches[x] = Branch(query_node, added_node)

    # ------------------------------------------------------------------
    # Private: acceptance ratio
    # ------------------------------------------------------------------

    def _acceptance_ratio(self, arg: "ARG") -> float:
        """Compute Metropolis acceptance ratio.

        Mirrors Threader_smc::acceptance_ratio.
        """
        # Height of the cut tree: max time among CHILD nodes (keys of parents dict).
        # C++: cut_tree.parents.rbegin()->first->time (rbegin = max-time key).
        # Root is never a key (it has no parent), so we never get time=inf here.
        cut_height = max(
            (child.time for child in arg.cut_tree.parents.keys()),
            default=0.0,
        )
        old_height = cut_height
        new_height = cut_height

        # Find old joining branch at cut_pos
        old_join_keys = [k for k in arg.joining_branches.keys() if k <= arg.cut_pos]
        old_join_branch = arg.joining_branches[max(old_join_keys)] if old_join_keys else None

        new_join_keys = [k for k in self.new_joining_branches.keys() if k <= arg.cut_pos]
        new_join_branch = self.new_joining_branches[max(new_join_keys)] if new_join_keys else None

        old_add_keys = [k for k in arg.removed_branches.keys() if k <= arg.cut_pos]
        old_add_branch = arg.removed_branches[max(old_add_keys)] if old_add_keys else None

        new_add_keys = [k for k in self.added_branches.keys() if k <= arg.cut_pos]
        new_add_branch = self.added_branches[max(new_add_keys)] if new_add_keys else None

        if old_join_branch is not None and old_join_branch.upper_node is arg.root:
            old_height = old_add_branch.upper_node.time if old_add_branch else cut_height
        if new_join_branch is not None and new_join_branch.upper_node is arg.root:
            new_height = new_add_branch.upper_node.time if new_add_branch else cut_height

        if new_height <= 0:
            return 1.0
        return old_height / new_height
