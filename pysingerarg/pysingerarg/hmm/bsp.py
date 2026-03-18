"""
BSP — Branch Sequence Propagator, the forward HMM for branch threading.

Mirrors BSP_smc.cpp / BSP_smc.hpp.

The BSP computes the HMM forward probabilities over a set of Interval
objects (each representing a (branch, time) cell).  At each genomic
position the HMM:
  1. Optionally applies a Recombination transfer (moves probability mass
     from old intervals to new ones according to the topology change).
  2. Advances by one bin (forward step), mixing staying-in-place with
     recombining to a new time.
  3. Multiplies by the emission probability (null or with mutations).

After the forward pass, sample_joining_branches() performs a traceback
to return a map of pos → Branch for threading.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from sortedcontainers import SortedDict

from ..data.branch import Branch
from ..data.interval import Interval, IntervalInfo
from .coalescent import CoalescentCalculator

if TYPE_CHECKING:
    from ..data.node import Node
    from ..data.recombination import Recombination
    from .emission import Emission


class BSP:
    """Branch Sequence Propagator — forward HMM for the branch dimension.

    State space: a list of Interval objects.
    Forward probabilities: forward_probs[step][interval_idx].

    Mirrors BSP_smc in the C++ code.
    """

    def __init__(self) -> None:
        self.cut_time: float = 0.0
        self.cutoff: float = 0.0
        self.check_points: Set[float] = set()
        self.eh: Optional["Emission"] = None

        self.curr_index: int = 0
        self.curr_intervals: List[Interval] = []
        self.valid_branches: Set[Branch] = set()

        # forward_probs[step] is a list of floats, one per interval
        self.forward_probs: List[List[float]] = []
        # state_spaces: step_idx → list of Intervals at that step
        self.state_spaces: SortedDict = SortedDict()  # int → List[Interval]

        self.recomb_probs: List[float] = []
        self.recomb_weights: List[float] = []
        self.null_emit_probs: List[float] = []
        self.mut_emit_probs: List[float] = []

        self.rhos: List[float] = []
        self.recomb_sums: List[float] = []
        self.weight_sums: List[float] = []
        self.recomb_sum: float = 0.0
        self.weight_sum: float = 0.0

        self.dim: int = 0
        self.states_change: bool = False
        self.sample_index: int = 0

        self.cc: CoalescentCalculator = CoalescentCalculator(0.0)

        self.prev_rho: float = -1.0
        self.prev_theta: float = -1.0
        self.prev_node: Optional["Node"] = None

        self._rng: np.random.Generator = np.random.default_rng()

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def _random(self) -> float:
        return float(self._rng.uniform())

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def reserve_memory(self, length: int) -> None:
        self.forward_probs = []

    def set_cutoff(self, x: float) -> None:
        self.cutoff = x

    def set_emission(self, e: "Emission") -> None:
        self.eh = e

    def set_check_points(self, p: Set[float]) -> None:
        self.check_points = p

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def start(self, branches: Set[Branch], t: float) -> None:
        """Initialise the forward pass at the left boundary.

        *branches* is the set of branches in the starting tree.
        *t* is the cut time (lineage starts at time t).

        Mirrors BSP_smc::start.
        """
        self.cut_time = t
        self.curr_index = 0
        self.valid_branches = set()
        self.curr_intervals = []
        self.rhos = []
        self.recomb_sums = []
        self.weight_sums = []
        self.state_spaces = SortedDict()

        for b in branches:
            if b.upper_node.time > t:
                self.valid_branches.add(b)

        self.cc = CoalescentCalculator(t)
        self.cc.compute(self.valid_branches)

        temp: List[float] = []
        for b in sorted(self.valid_branches, key=lambda x: x):
            lb = max(b.lower_node.time, t)
            ub = b.upper_node.time
            p = self.cc.weight(lb, ub)
            interval = Interval(b, lb, ub, self.curr_index)
            self.curr_intervals.append(interval)
            temp.append(p)

        self.forward_probs.append(temp)
        self._compute_interval_info()
        self.weight_sums.append(0.0)
        self._set_dimensions()
        self.state_spaces[self.curr_index] = list(self.curr_intervals)
        self.states_change = False

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def forward(self, rho: float) -> None:
        """Advance forward by one bin with recombination rate *rho*.

        Mirrors BSP_smc::forward.
        """
        self.rhos.append(rho)
        self._compute_recomb_probs(rho)
        self._compute_recomb_weights(rho)
        self.prev_rho = rho
        self.curr_index += 1

        # recomb_sum = sum_i(recomb_probs[i] * forward_probs[curr-1][i])
        prev_fp = self.forward_probs[self.curr_index - 1]
        self.recomb_sum = sum(
            self.recomb_probs[i] * prev_fp[i] for i in range(self.dim)
        )

        new_fp = [0.0] * self.dim
        for i in range(self.dim):
            new_fp[i] = (
                prev_fp[i] * (1.0 - self.recomb_probs[i])
                + self.recomb_sum * self.recomb_weights[i]
            )
        self.forward_probs.append(new_fp)
        self.recomb_sums.append(self.recomb_sum)
        self.weight_sums.append(self.weight_sum)

    # ------------------------------------------------------------------
    # Transfer at recombination
    # ------------------------------------------------------------------

    def transfer(self, r: "Recombination") -> None:
        """Apply topology change *r* to the interval state space.

        Mirrors BSP_smc::transfer.
        """
        self.rhos.append(0.0)
        self.prev_rho = -1.0
        self.prev_theta = -1.0
        self.recomb_sums.append(0.0)
        self.weight_sums.append(0.0)

        self.sanity_check(r)
        self.curr_index += 1

        transfer_weights: Dict[IntervalInfo, List[float]] = {}
        transfer_intervals: Dict[IntervalInfo, List[Interval]] = {}

        self._update_states(r.deleted_branches, r.inserted_branches)

        for i in range(len(self.curr_intervals)):
            self._process_interval(r, i, transfer_weights, transfer_intervals)

        self._add_new_branches(r, transfer_weights, transfer_intervals)
        self._generate_intervals(r, transfer_weights, transfer_intervals)
        self._set_dimensions()
        self.state_spaces[self.curr_index] = list(self.curr_intervals)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def null_emit(self, theta: float, query_node: Optional["Node"]) -> None:
        """Apply null-emission (no mutations in bin).

        Mirrors BSP_smc::null_emit.
        """
        self._compute_null_emit_prob(theta, query_node)
        self.prev_theta = theta
        self.prev_node = query_node
        fp = self.forward_probs[self.curr_index]
        ws = 0.0
        for i in range(self.dim):
            fp[i] *= self.null_emit_probs[i]
            ws += fp[i]
        if ws <= 0:
            raise RuntimeError("BSP null_emit: forward prob sum is zero")
        for i in range(self.dim):
            fp[i] /= ws

    def mut_emit(
        self,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        query_node: Optional["Node"],
    ) -> None:
        """Apply mutation emission.

        Mirrors BSP_smc::mut_emit.
        """
        self._compute_mut_emit_probs(theta, bin_size, mut_set, query_node)
        fp = self.forward_probs[self.curr_index]
        ws = 0.0
        for i in range(self.dim):
            fp[i] *= self.mut_emit_probs[i]
            ws += fp[i]
        if ws <= 0:
            raise RuntimeError("BSP mut_emit: forward prob sum is zero")
        for i in range(self.dim):
            fp[i] /= ws

    def sanity_check(self, r: "Recombination") -> None:
        """Zero out invalid point-mass intervals at recombination nodes.

        Mirrors BSP_smc::sanity_check.
        """
        for i, interval in enumerate(self.curr_intervals):
            if (interval.lb == interval.ub
                    and interval.lb == r.inserted_node.time
                    and interval.branch != r.target_branch):
                self.forward_probs[self.curr_index][i] = 0.0

    # ------------------------------------------------------------------
    # Traceback / sampling
    # ------------------------------------------------------------------

    def sample_joining_branches(
        self,
        start_index: int,
        coordinates: List[float],
    ) -> SortedDict:
        """Traceback to sample a joining-branch map.

        Mirrors BSP_smc::sample_joining_branches.
        Returns SortedDict[pos → Branch].
        """
        self.prev_rho = -1.0
        joining_branches: SortedDict = SortedDict()
        x = self.curr_index

        pos = coordinates[x + start_index + 1]
        interval = self._sample_curr_interval(x)
        joining_branches[pos] = interval.branch

        while x >= 0:
            x = self._trace_back_helper(interval, x)
            pos = coordinates[x + start_index]
            joining_branches[pos] = interval.branch

            if x == 0:
                break
            elif x == interval.start_pos:
                x -= 1
                interval = self._sample_source_interval(interval, x)
            else:
                x -= 1
                interval = self._sample_prev_interval(x)

        self._simplify(joining_branches)
        return joining_branches

    def avg_num_states(self) -> float:
        """Average number of states (intervals) per position."""
        if len(self.state_spaces) <= 1:
            return 0.0
        span = 0
        count = 0.0
        keys = list(self.state_spaces.keys())
        for i in range(1, len(keys)):
            if keys[i] == _INT_MAX_IDX:
                break
            count += len(self.state_spaces[keys[i]]) * (keys[i] - keys[i - 1])
            span = keys[i]
        return count / span if span > 0 else 0.0

    # ------------------------------------------------------------------
    # Private helpers: state management
    # ------------------------------------------------------------------

    def _update_states(
        self,
        deletions: Set[Branch],
        insertions: Set[Branch],
    ) -> None:
        for b in deletions:
            if b.upper_node.time > self.cut_time:
                self.valid_branches.discard(b)
            self.states_change = True
        for b in insertions:
            if b.upper_node.time > self.cut_time:
                self.valid_branches.add(b)
            self.states_change = True

    def _set_dimensions(self) -> None:
        self.dim = len(self.curr_intervals)
        self.recomb_probs = [0.0] * self.dim
        self.recomb_weights = [0.0] * self.dim
        self.null_emit_probs = [0.0] * self.dim
        self.mut_emit_probs = [0.0] * self.dim

    def _compute_interval_info(self) -> None:
        """Update weight and representative time for each current interval."""
        if self.states_change:
            self.cc.compute(self.valid_branches)
        self.states_change = False
        for interval in self.curr_intervals:
            p = self.cc.weight(interval.lb, interval.ub)
            t = self.cc.time(interval.lb, interval.ub)
            interval.assign_weight(p)
            interval.assign_time(t)

    def _get_recomb_prob(self, rho: float, t: float) -> float:
        """P(recombination | rho, t) = rho*(t-cut_time)*exp(-rho*(t-cut_time))."""
        dt = t - self.cut_time
        return rho * dt * math.exp(-rho * dt)

    def _compute_recomb_probs(self, rho: float) -> None:
        if rho == self.prev_rho:
            return
        for i, interval in enumerate(self.curr_intervals):
            self.recomb_probs[i] = self._get_recomb_prob(rho, interval.time)

    def _compute_recomb_weights(self, rho: float) -> None:
        if rho == self.prev_rho:
            return
        for i, interval in enumerate(self.curr_intervals):
            if interval.full(self.cut_time):
                self.recomb_weights[i] = self.recomb_probs[i] * interval.weight
            else:
                self.recomb_weights[i] = 0.0
        ws = sum(self.recomb_weights)
        self.weight_sum = ws
        if ws > 0:
            for i in range(self.dim):
                self.recomb_weights[i] /= ws

    def _compute_null_emit_prob(
        self, theta: float, query_node: Optional["Node"]
    ) -> None:
        if theta == self.prev_theta and query_node is self.prev_node:
            return
        for i, interval in enumerate(self.curr_intervals):
            self.null_emit_probs[i] = self.eh.null_emit(
                interval.branch, interval.time, theta, query_node
            )

    def _compute_mut_emit_probs(
        self,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        query_node: Optional["Node"],
    ) -> None:
        for i, interval in enumerate(self.curr_intervals):
            self.mut_emit_probs[i] = self.eh.mut_emit(
                interval.branch, interval.time, theta, bin_size, mut_set, query_node
            )

    # ------------------------------------------------------------------
    # Private helpers: transfer
    # ------------------------------------------------------------------

    def _add_new_branches(
        self,
        r: "Recombination",
        tw: Dict[IntervalInfo, List[float]],
        ti: Dict[IntervalInfo, List[Interval]],
    ) -> None:
        """Add recombined and merging branches to transfer maps."""
        if (r.merging_branch
                and r.merging_branch.lower_node is not None
                and r.merging_branch.upper_node is not None
                and r.merging_branch.upper_node.time > self.cut_time):
            lb = max(self.cut_time, r.merging_branch.lower_node.time)
            ub = r.merging_branch.upper_node.time
            if lb <= ub:
                key = IntervalInfo(r.merging_branch, lb, ub)
                if key not in tw:
                    tw[key] = []
                    ti[key] = []

        if (r.recombined_branch
                and r.recombined_branch.lower_node is not None
                and r.recombined_branch.upper_node is not None
                and r.recombined_branch.upper_node.time > self.cut_time):
            lb = max(self.cut_time, r.recombined_branch.lower_node.time)
            ub = r.recombined_branch.upper_node.time
            if lb <= ub:
                key = IntervalInfo(r.recombined_branch, lb, ub)
                if key not in tw:
                    tw[key] = []
                    ti[key] = []

    def _transfer_helper(
        self,
        key: IntervalInfo,
        prev_interval: Optional[Interval],
        w: float,
        tw: Dict[IntervalInfo, List[float]],
        ti: Dict[IntervalInfo, List[Interval]],
    ) -> None:
        if key not in tw:
            tw[key] = []
            ti[key] = []
        if prev_interval is not None:
            tw[key].append(w)
            ti[key].append(prev_interval)

    def _process_interval(
        self,
        r: "Recombination",
        i: int,
        tw: Dict[IntervalInfo, List[float]],
        ti: Dict[IntervalInfo, List[Interval]],
    ) -> None:
        b = self.curr_intervals[i].branch
        if b == r.source_branch:
            self._process_source_interval(r, i, tw, ti)
        elif b == r.target_branch:
            self._process_target_interval(r, i, tw, ti)
        else:
            self._process_other_interval(r, i, tw, ti)

    def _process_source_interval(
        self, r: "Recombination", i: int,
        tw: Dict, ti: Dict,
    ) -> None:
        prev = self.curr_intervals[i]
        p = self.forward_probs[self.curr_index - 1][i]
        break_time = r.start_time
        point_time = r.source_branch.upper_node.time

        if prev.ub <= break_time:
            key = IntervalInfo(r.recombined_branch, prev.lb, prev.ub)
            self._transfer_helper(key, prev, p, tw, ti)
        elif prev.lb >= break_time:
            key = IntervalInfo(r.merging_branch, point_time, point_time)
            self._transfer_helper(key, prev, p, tw, ti)
        else:
            w1 = self.cc.weight(prev.lb, break_time)
            w2 = self.cc.weight(break_time, prev.ub)
            if w1 == 0 and w2 == 0:
                w1, w2 = 1.0, 0.0
            else:
                total = w1 + w2
                w1 = w1 / total
                w2 = 1.0 - w1
            key1 = IntervalInfo(r.recombined_branch, prev.lb, break_time)
            self._transfer_helper(key1, prev, w1 * p, tw, ti)
            key2 = IntervalInfo(r.merging_branch, point_time, point_time)
            self._transfer_helper(key2, prev, w2 * p, tw, ti)

    def _process_target_interval(
        self, r: "Recombination", i: int,
        tw: Dict, ti: Dict,
    ) -> None:
        prev = self.curr_intervals[i]
        p = self.forward_probs[self.curr_index - 1][i]
        join_time = r.inserted_node.time

        if prev.lb == prev.ub == join_time:
            lb = max(self.cut_time, r.start_time)
            ub = r.recombined_branch.upper_node.time
            if lb <= ub:
                key = IntervalInfo(r.recombined_branch, lb, ub)
                self._transfer_helper(key, prev, p, tw, ti)
        elif prev.lb >= join_time:
            key = IntervalInfo(r.upper_transfer_branch, prev.lb, prev.ub)
            self._transfer_helper(key, prev, p, tw, ti)
        elif prev.ub <= join_time:
            key = IntervalInfo(r.lower_transfer_branch, prev.lb, prev.ub)
            self._transfer_helper(key, prev, p, tw, ti)
        else:
            w0 = self._get_overwrite_prob(r, prev.lb, prev.ub)
            w1 = self.cc.weight(prev.lb, join_time)
            w2 = self.cc.weight(join_time, prev.ub)
            if w1 + w2 == 0:
                w1 = w2 = 0.0
                w0 = 1.0
            else:
                total = w1 + w2
                w1 = w1 / total
                w2 = 1.0 - w1
                w1 *= (1.0 - w0)
                w2 *= (1.0 - w0)
            key1 = IntervalInfo(r.lower_transfer_branch, prev.lb, join_time)
            self._transfer_helper(key1, prev, w1 * p, tw, ti)
            key2 = IntervalInfo(r.upper_transfer_branch, join_time, prev.ub)
            self._transfer_helper(key2, prev, w2 * p, tw, ti)
            lb = max(r.start_time, self.cut_time)
            ub = r.recombined_branch.upper_node.time
            if lb <= ub:
                key3 = IntervalInfo(r.recombined_branch, lb, ub)
                self._transfer_helper(key3, prev, w0 * p, tw, ti)

    def _process_other_interval(
        self, r: "Recombination", i: int,
        tw: Dict, ti: Dict,
    ) -> None:
        prev = self.curr_intervals[i]
        p = self.forward_probs[self.curr_index - 1][i]
        if r.affect(prev.branch):
            key = IntervalInfo(r.merging_branch, prev.lb, prev.ub)
        else:
            key = IntervalInfo(prev.branch, prev.lb, prev.ub)
        self._transfer_helper(key, prev, p, tw, ti)

    def _get_overwrite_prob(self, r: "Recombination", lb: float, ub: float) -> float:
        if r.pos in self.check_points:
            return 0.0
        join_time = r.inserted_node.time
        p1 = self.cc.weight(lb, ub)
        p2 = self.cc.weight(max(self.cut_time, r.start_time), join_time)
        if p1 == 0 and p2 == 0:
            return 1.0
        return p2 / (p1 + p2)

    def _generate_intervals(
        self,
        r: "Recombination",
        tw: Dict[IntervalInfo, List[float]],
        ti: Dict[IntervalInfo, List[Interval]],
    ) -> None:
        """Build new curr_intervals from transfer maps."""
        new_intervals: List[Interval] = []
        new_fp: List[float] = []

        for key, weights in tw.items():
            intervals_src = ti[key]
            b = key.branch
            if b.lower_node is None or b.upper_node is None:
                continue
            lb = key.lb
            ub = key.ub
            p = sum(weights)

            full_lb = max(self.cut_time, b.lower_node.time)
            full_ub = b.upper_node.time
            is_full = (lb == full_lb and ub == full_ub)

            if is_full:
                new_iv = Interval(b, lb, ub, self.curr_index)
                new_intervals.append(new_iv)
                new_fp.append(p)
                if weights:
                    new_iv.source_weights = list(weights)
                    new_iv.source_intervals = list(intervals_src)
            elif p >= self.cutoff:
                new_iv = Interval(b, lb, ub, self.curr_index)
                new_intervals.append(new_iv)
                new_fp.append(p)
                if weights:
                    new_iv.source_weights = list(weights)
                    new_iv.source_intervals = list(intervals_src)

        self.forward_probs.append(new_fp)
        self.curr_intervals = new_intervals
        self._compute_interval_info()

    # ------------------------------------------------------------------
    # Private helpers: state-space lookup
    # ------------------------------------------------------------------

    def _get_state_space(self, x: int) -> List[Interval]:
        idx = self.state_spaces.bisect_right(x) - 1
        if idx < 0:
            return self.curr_intervals
        key = self.state_spaces.keys()[idx]
        return self.state_spaces[key]

    def _get_prev_breakpoint(self, x: int) -> int:
        idx = self.state_spaces.bisect_right(x) - 1
        if idx < 0:
            return 0
        return self.state_spaces.keys()[idx]

    # ------------------------------------------------------------------
    # Private helpers: traceback sampling
    # ------------------------------------------------------------------

    def _sample_curr_interval(self, x: int) -> Interval:
        intervals = self._get_state_space(x)
        fp = self.forward_probs[x]
        ws = sum(fp)
        q = self._random()
        w = ws * q
        for i, interval in enumerate(intervals):
            w -= fp[i]
            if w <= 0:
                self.sample_index = i
                return interval
        # Fallback
        self.sample_index = len(intervals) - 1
        return intervals[-1]

    def _sample_prev_interval(self, x: int) -> Interval:
        intervals = self._get_state_space(x)
        rho = self.rhos[x]
        ws = self.recomb_sums[x]
        q = self._random()
        w = ws * q
        for i, interval in enumerate(intervals):
            contrib = self._get_recomb_prob(rho, interval.time) * self.forward_probs[x][i]
            w -= contrib
            if w <= 0:
                self.sample_index = i
                return interval
        self.sample_index = len(intervals) - 1
        return intervals[-1]

    def _sample_source_interval(self, interval: Interval, x: int) -> Interval:
        prev_intervals = self._get_state_space(x)
        weights = interval.source_weights
        sources = interval.source_intervals
        ws = sum(weights)
        q = self._random()
        w = ws * q
        for i, src in enumerate(sources):
            w -= weights[i]
            if w <= 0:
                try:
                    self.sample_index = prev_intervals.index(src)
                except ValueError:
                    self.sample_index = 0
                return src
        src = sources[-1]
        try:
            self.sample_index = prev_intervals.index(src)
        except ValueError:
            self.sample_index = 0
        return src

    def _trace_back_helper(self, interval: Interval, x: int) -> int:
        """Walk backward from step x, deciding when to jump to a new lineage.

        Mirrors BSP_smc::trace_back_helper.
        """
        if not interval.full(self.cut_time):
            return interval.start_pos

        p = self._random()
        q = 1.0
        while x > interval.start_pos:
            recomb_sum = self.recomb_sums[x - 1]
            weight_sum = self.weight_sums[x]
            if recomb_sum == 0:
                shrinkage = 1.0
            else:
                rp = self._get_recomb_prob(self.rhos[x - 1], interval.time)
                non_recomb_prob = (1.0 - rp) * self.forward_probs[x - 1][self.sample_index]
                all_prob = non_recomb_prob + recomb_sum * interval.weight * rp / (weight_sum if weight_sum > 0 else 1.0)
                if all_prob <= 0:
                    shrinkage = 1.0
                else:
                    shrinkage = non_recomb_prob / all_prob
                    shrinkage = max(0.0, min(1.0, shrinkage))
            q *= shrinkage
            if p >= q:
                return x
            x -= 1
        return interval.start_pos

    @staticmethod
    def _simplify(joining_branches: SortedDict) -> None:
        """Deduplicate consecutive identical branches in the joining-branch map."""
        if len(joining_branches) <= 1:
            return
        keys = list(joining_branches.keys())
        simplified: SortedDict = SortedDict()
        curr = joining_branches[keys[0]]
        simplified[keys[0]] = curr
        for k in keys[1:]:
            if joining_branches[k] != curr:
                simplified[k] = joining_branches[k]
                curr = joining_branches[k]
        # Always keep the last entry
        simplified[keys[-1]] = joining_branches[keys[-1]]
        joining_branches.clear()
        for k, v in simplified.items():
            joining_branches[k] = v


# Sentinel used in avg_num_states
_INT_MAX_IDX = 10**18
