"""
TSP — Time Sequence Propagator, the forward HMM for coalescence times.

Mirrors TSP_smc.cpp / TSP_smc.hpp.

The TSP operates on a single Branch and samples a representative
coalescence time in each genomic bin.  It uses a PSMC-style transition
kernel (psmc_prob) rather than the coalescent-CDF-based kernel of BSP.

Key public API
--------------
start(branch, t)                     — initialise at the left boundary
forward(rho)                         — advance one bin
transfer(r, prev_branch, next_branch)— apply topology change
recombine(prev_branch, next_branch)  — re-sample at a recombination
null_emit(theta, query_node)         — apply no-mutation emission
mut_emit(theta, bin_size, mut_set, query_node) — apply mutation emission
sample_joining_nodes(start_index, coordinates) — traceback → Dict[pos, Node]
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from sortedcontainers import SortedDict

from ..data.branch import Branch
from ..data.interval import Interval
from ..data.node import Node

if TYPE_CHECKING:
    from ..data.recombination import Recombination
    from .emission import Emission


_EPSILON = 1e-20
_counter: int = 0  # mirrors TSP_smc::counter (static)


def _new_node(t: float) -> Node:
    """Create an unnamed node at time *t* (mirrors new_node helper)."""
    n = Node(time=t)
    return n


class TSP:
    """Time Sequence Propagator — forward HMM for the time dimension.

    State space: a list of Interval objects (finely gridded over a Branch).
    Forward probabilities: forward_probs[step][interval_idx].

    Mirrors TSP_smc in the C++ code.
    """

    def __init__(self) -> None:
        global _counter
        self.cut_time: float = 0.0
        self.lower_bound: float = 0.0
        self.gap: float = 0.02          # default quantile gap for grid generation
        self.eh: Optional["Emission"] = None
        self.check_points: Set[float] = set()

        self.curr_index: int = 0
        self.curr_branch: Optional[Branch] = None
        self.curr_intervals: List[Interval] = []

        self.forward_probs: List[List[float]] = []
        self.state_spaces: Dict[int, List[Interval]] = {}   # int → List[Interval]
        self.source_interval: Dict[int, Interval] = {}      # id(new_iv) → old Interval

        self.rhos: List[float] = []

        # per-step work arrays (resized in set_dimensions)
        self.dim: int = 0
        self.diagonals: List[float] = []
        self.lower_diagonals: List[float] = []
        self.upper_diagonals: List[float] = []
        self.lower_sums: List[float] = []
        self.upper_sums: List[float] = []
        self.null_emit_probs: List[float] = []
        self.mut_emit_probs: List[float] = []
        self.factors: List[float] = []
        self.trace_back_probs: List[float] = []
        self.emissions: List[float] = [0.0, 0.0, 0.0, 0.0]

        self.sample_index: int = 0
        self.prev_rho: float = -1.0
        self.prev_theta: float = -1.0
        self.prev_node: Optional[Node] = None

        # temp buffer for building forward_probs at current step
        self._temp: List[float] = []

        self._rng: np.random.Generator = np.random.default_rng()

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def _random(self) -> float:
        return float(self._rng.uniform())

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_gap(self, q: float) -> None:
        self.gap = q

    def set_emission(self, e: "Emission") -> None:
        self.eh = e

    def set_check_points(self, p: Set[float]) -> None:
        self.check_points = p

    def reserve_memory(self, length: int) -> None:
        self.forward_probs = []
        self.rhos = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def start(self, branch: Branch, t: float) -> None:
        """Initialise forward pass at the left boundary.

        Mirrors TSP_smc::start.
        """
        self.cut_time = t
        self.curr_index = 0
        self.curr_branch = branch
        self._temp = []
        self.curr_intervals = []
        self.source_interval = {}
        self.state_spaces = {}
        self.forward_probs = []
        self.rhos = []
        self.prev_rho = -1.0
        self.prev_theta = -1.0
        self.prev_node = None

        lb_start = branch.lower_node.time
        ub_start = branch.upper_node.time
        self.lower_bound = max(t, lb_start)

        self._generate_intervals(branch, lb_start, ub_start)

        # initial forward probs: exp(-lb) - exp(-ub) for each interval
        for iv in self.curr_intervals:
            self._temp.append(math.exp(-iv.lb) - math.exp(-iv.ub))

        self.state_spaces[0] = list(self.curr_intervals)
        self.forward_probs.append(list(self._temp))
        self._temp = []
        self._set_dimensions()
        self._compute_factors()

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def forward(self, rho: float) -> None:
        """Advance one bin.

        Mirrors TSP_smc::forward.
        """
        self.rhos.append(rho)
        if self.dim == 0:
            self.curr_index += 1
            self.prev_rho = rho
            self.forward_probs.append([])
            return
        self._compute_diagonals(rho)
        self._compute_lower_diagonals(rho)
        self._compute_upper_diagonals(rho)
        self._compute_lower_sums()
        self._compute_upper_sums()
        self.curr_index += 1
        self.prev_rho = rho

        new_fp = list(self.lower_sums)  # copy
        for i in range(self.dim):
            new_fp[i] += (
                self.diagonals[i] * self.forward_probs[self.curr_index - 1][i]
                + self.lower_diagonals[i] * self.upper_sums[i]
            )
            if self.curr_intervals[i].lb != self.curr_intervals[i].ub:
                new_fp[i] = max(_EPSILON, new_fp[i])
        self.forward_probs.append(new_fp)

    # ------------------------------------------------------------------
    # Transfer at topology change
    # ------------------------------------------------------------------

    def transfer(
        self,
        r: "Recombination",
        prev_branch: Branch,
        next_branch: Branch,
    ) -> None:
        """Apply topology change *r*.

        Mirrors TSP_smc::transfer.
        """
        self.rhos.append(0.0)
        self.prev_rho = -1.0
        self.prev_theta = -1.0
        self.prev_node = None
        self._sanity_check(r)
        self.curr_index += 1
        self.curr_branch = next_branch
        self.lower_bound = max(self.cut_time, next_branch.lower_node.time)

        # Constrain previous step's probs based on topology
        if prev_branch == r.source_branch and next_branch == r.merging_branch:
            self._set_interval_constraint(r)
        elif prev_branch == r.target_branch and next_branch == r.recombined_branch:
            self._set_point_constraint(r)

        self.curr_intervals = []
        self._temp = []

        if prev_branch == r.source_branch and next_branch == r.merging_branch:
            # Switch to a point mass at deleted_node.time
            t = r.deleted_node.time
            # Clamp to branch range so rescaling can't produce out-of-bounds t
            lb_b = next_branch.lower_node.time
            ub_b = next_branch.upper_node.time
            t = max(lb_b, min(ub_b, t))
            self._generate_intervals(next_branch, lb_b, t)
            n_before = len(self.curr_intervals)
            self._generate_intervals(next_branch, t, t)
            # Mark point interval as point mass (only if one was actually added)
            if len(self.curr_intervals) > n_before:
                self._temp[-1] = 1.0
                self.curr_intervals[-1].node = r.deleted_node
            self._generate_intervals(next_branch, t, ub_b)

        elif prev_branch == r.target_branch and next_branch == r.recombined_branch:
            # Switch from a point mass
            self._generate_intervals(
                next_branch, next_branch.lower_node.time, r.start_time
            )
            self._generate_intervals(
                next_branch, r.start_time, next_branch.upper_node.time
            )
            for i, iv in enumerate(self.curr_intervals):
                if iv.time >= r.start_time:
                    self._temp[i] = 1.0

        else:
            # Regular transfer: overlap of prev and next branch intervals
            lb = next_branch.lower_node.time
            ub = max(prev_branch.lower_node.time, next_branch.lower_node.time)
            self._generate_intervals(next_branch, lb, ub)
            self._transfer_intervals(r, prev_branch, next_branch)
            if self.curr_intervals:
                lb2 = min(self.curr_intervals[-1].ub, next_branch.upper_node.time)
            else:
                lb2 = next_branch.lower_node.time
            ub2 = next_branch.upper_node.time
            self._generate_intervals(next_branch, lb2, ub2)

        self.state_spaces[self.curr_index] = list(self.curr_intervals)
        self.forward_probs.append(list(self._temp))
        self._temp = []
        self._set_dimensions()
        self._compute_factors()

    # ------------------------------------------------------------------
    # Recombination (full re-sample)
    # ------------------------------------------------------------------

    def recombine(self, prev_branch: Branch, next_branch: Branch) -> None:
        """Re-sample interval distribution at a recombination.

        Mirrors TSP_smc::recombine.
        """
        prev_intervals = list(self.curr_intervals)
        prev_fp = list(self.forward_probs[self.curr_index])

        self.curr_intervals = []
        self._temp = []
        self.rhos.append(0.0)
        self.prev_rho = -1.0
        self.prev_theta = -1.0
        self.prev_node = None
        self.curr_branch = next_branch
        self.curr_index += 1
        self.lower_bound = max(self.cut_time, next_branch.lower_node.time)

        self._generate_intervals(
            next_branch,
            next_branch.lower_node.time,
            next_branch.upper_node.time,
        )
        # Initialize new step fp to zero, then accumulate
        new_fp = [0.0] * len(self.curr_intervals)
        self.forward_probs.append(new_fp)
        self.state_spaces[self.curr_index] = list(self.curr_intervals)
        self._set_dimensions()
        self._compute_factors()

        lb_full = self.curr_intervals[0].lb
        ub_full = self.curr_intervals[-1].ub

        for i, prev_iv in enumerate(prev_intervals):
            base = self._recomb_prob(prev_iv.time, lb_full, ub_full)
            for j, curr_iv in enumerate(self.curr_intervals):
                if base == 0:
                    new_prob = 1.0
                else:
                    new_prob = (
                        self._recomb_prob(prev_iv.time, curr_iv.lb, curr_iv.ub)
                        * prev_fp[i]
                        / base
                    )
                new_fp[j] += new_prob + _EPSILON

        self._temp = []

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def null_emit(self, theta: float, query_node: "Node") -> None:
        """Apply null emission (no mutations).

        Mirrors TSP_smc::null_emit.
        """
        if self.dim == 0:
            return
        self._compute_null_emit_probs(theta, query_node)
        self.prev_theta = theta
        self.prev_node = query_node
        fp = self.forward_probs[self.curr_index]
        ws = 0.0
        for i in range(self.dim):
            fp[i] *= self.null_emit_probs[i]
            ws += fp[i]
        if ws > 0:
            for i in range(self.dim):
                fp[i] /= ws
        else:
            for i in range(self.dim):
                fp[i] = 1.0 / self.dim

    def mut_emit(
        self,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        query_node: "Node",
    ) -> None:
        """Apply mutation emission.

        Mirrors TSP_smc::mut_emit.
        """
        if self.dim == 0:
            return
        self._compute_mut_emit_probs(theta, bin_size, mut_set, query_node)
        fp = self.forward_probs[self.curr_index]
        ws = 0.0
        for i in range(self.dim):
            fp[i] *= self.mut_emit_probs[i]
            ws += fp[i]
        if ws > 0:
            for i in range(self.dim):
                fp[i] /= ws
        else:
            for i in range(self.dim):
                fp[i] = 1.0 / self.dim if self.dim > 0 else 0.0

    # ------------------------------------------------------------------
    # Traceback
    # ------------------------------------------------------------------

    def sample_joining_nodes(
        self,
        start_index: int,
        coordinates: List[float],
    ) -> Dict[float, Optional[Node]]:
        """Traceback to sample a joining-node map.

        Returns Dict[pos → Node].  Mirrors TSP_smc::sample_joining_nodes.
        """
        self.prev_rho = -1.0
        joining_nodes: Dict[float, Optional[Node]] = {}

        x = self.curr_index
        pos = coordinates[x + start_index + 1]
        interval = self._sample_curr_interval(x)
        n = self._sample_joining_node(interval)
        joining_nodes[pos] = None  # sentinel for rightmost pos

        while x >= 0:
            x = self._trace_back_helper(interval, x)
            pos = coordinates[x + start_index]
            joining_nodes[pos] = n

            if x == 0:
                break
            if x == interval.start_pos:
                if id(interval) in self.source_interval:
                    x -= 1
                    interval = self._sample_source_interval(interval, x)
                else:
                    x -= 1
                    interval = self._sample_recomb_interval(interval, x)
                    n = self._sample_joining_node(interval)
            else:
                x -= 1
                interval = self._sample_prev_interval(interval, x)
                n = self._sample_joining_node(interval)

            self.prev_rho = -1.0

        return joining_nodes

    # ------------------------------------------------------------------
    # Private: grid and interval generation
    # ------------------------------------------------------------------

    def _get_exp_quantile(self, p: float) -> float:
        """Inverse CDF of Exp(1): -log(1-p).  Mirrors TSP_smc::get_exp_quantile."""
        if p < 1e-6:
            return 0.0
        if 1.0 - p < 1e-6:
            return math.inf
        return -math.log(1.0 - p)

    def _generate_grid(self, lb: float, ub: float) -> List[float]:
        """Generate a quantile-spaced grid in [lb, ub].

        Mirrors TSP_smc::generate_grid.
        """
        lq = 1.0 - math.exp(-lb)
        uq = 1.0 - math.exp(-ub)
        q = uq - lq
        n = math.ceil(q / self.gap)
        points = [lb]
        for i in range(1, n):
            l = self._get_exp_quantile(lq + i * q / n)
            points.append(l)
        points.append(ub)
        return points

    def _generate_intervals(
        self, branch: Branch, lb: float, ub: float
    ) -> None:
        """Append fine-grid intervals for [lb, ub] on *branch*.

        Mirrors TSP_smc::generate_intervals.
        """
        lb = max(self.cut_time, lb)
        ub = max(self.cut_time, ub)
        if lb > ub:
            return  # inverted bounds: mirrors C++ silently producing empty grid
        if lb == ub:
            # Point interval: skip boundary points
            lower_bound_iv = max(self.cut_time, branch.lower_node.time)
            upper_bound_iv = branch.upper_node.time
            if lb == lower_bound_iv or lb == upper_bound_iv:
                return
            iv = Interval(branch, lb, ub, self.curr_index)
            iv.fill_time()
            self.curr_intervals.append(iv)
            self._temp.append(0.0)
            return

        points = self._generate_grid(lb, ub)
        for i in range(len(points) - 1):
            l, u = points[i], points[i + 1]
            iv = Interval(branch, l, u, self.curr_index)
            iv.fill_time()
            self.curr_intervals.append(iv)
            self._temp.append(0.0)

    def _transfer_intervals(
        self,
        r: "Recombination",
        prev_branch: Branch,
        next_branch: Branch,
    ) -> None:
        """Transfer probability mass from prev to overlapping new intervals.

        Mirrors TSP_smc::transfer_intervals.
        """
        prev_intervals = self._get_state_space(self.curr_index - 1)
        for i, interval in enumerate(prev_intervals):
            lb = max(interval.lb, next_branch.lower_node.time)
            ub = min(interval.ub, next_branch.upper_node.time)
            if prev_branch is r.source_branch:
                ub = min(ub, r.start_time)
                if lb == r.start_time:
                    continue
            if lb == ub == next_branch.upper_node.time:
                continue
            if lb == ub == next_branch.lower_node.time:
                continue
            if ub >= lb:
                w = self._get_prop(lb, ub, interval.lb, interval.ub)
                p = w * self.forward_probs[self.curr_index - 1][i]
                new_iv = Interval(next_branch, lb, ub, self.curr_index)
                new_iv.fill_time()
                new_iv.node = interval.node
                self.source_interval[id(new_iv)] = interval
                self.curr_intervals.append(new_iv)
                self._temp.append(p)

    # ------------------------------------------------------------------
    # Private: dimension setup and transition factors
    # ------------------------------------------------------------------

    def _set_dimensions(self) -> None:
        self.dim = len(self.curr_intervals)
        self.diagonals = [0.0] * self.dim
        self.lower_diagonals = [0.0] * self.dim
        self.upper_diagonals = [0.0] * self.dim
        self.lower_sums = [0.0] * self.dim
        self.upper_sums = [0.0] * self.dim
        self.null_emit_probs = [0.0] * self.dim
        self.mut_emit_probs = [0.0] * self.dim
        self.factors = [0.0] * self.dim

    def _compute_factors(self) -> None:
        """Pre-compute ratio factors for lower_sums recursion.

        Mirrors TSP_smc::compute_factors.
        """
        if self.dim == 0:
            return
        self.factors[0] = 0.0
        for i in range(1, self.dim):
            iv_prev = self.curr_intervals[i - 1]
            iv_curr = self.curr_intervals[i]
            if iv_prev.lb == iv_prev.ub:
                self.factors[i] = 0.0
            elif iv_prev.ub - iv_prev.lb < 1e-4:
                self.factors[i] = 5.0
            else:
                num = math.exp(-iv_curr.lb) - math.exp(-iv_curr.ub)
                den = math.exp(-iv_prev.lb) - math.exp(-iv_prev.ub)
                self.factors[i] = min(num / den, 5.0) if den != 0 else 5.0

    # ------------------------------------------------------------------
    # Private: PSMC transition kernel
    # ------------------------------------------------------------------

    def _recomb_cdf(self, s: float, t: float) -> float:
        """Recombination CDF.  Mirrors TSP_smc::recomb_cdf."""
        if math.isinf(t):
            return 1.0
        if t == 0:
            return 0.0
        l = s - self.cut_time
        if s > t:
            cdf = t + math.expm1(self.cut_time - t) - self.cut_time
        else:
            cdf = s + math.expm1(self.cut_time - t) - math.expm1(s - t) - self.cut_time
        cdf = cdf / l
        return cdf

    def _recomb_prob(self, s: float, t1: float, t2: float) -> float:
        """P(recombination targets [t1, t2] | current time s).

        Mirrors TSP_smc::recomb_prob.
        """
        if s - max(self.lower_bound, self.cut_time) < 0.005:
            return math.exp(-t1) - math.exp(-t2)
        pl = self._recomb_cdf(s, t1)
        pu = self._recomb_cdf(s, t2)
        p = pu - pl
        p = max(p, 1e-5)
        return p

    def _psmc_cdf(self, rho: float, s: float, t: float) -> float:
        """PSMC CDF.  Mirrors TSP_smc::psmc_cdf."""
        l = 2.0 * s - self.lower_bound - self.cut_time
        if l == 0:
            pre_factor = rho
        else:
            pre_factor = (1.0 - math.exp(-rho * l)) / l
        if t == self.cut_time and t == self.lower_bound:
            return 0.0
        elif t <= s:
            integral = (
                2.0 * t
                + math.exp(-t) * (math.exp(self.cut_time) + math.exp(self.lower_bound))
                - self.cut_time - self.lower_bound - 2.0
            )
        else:
            integral = (
                2.0 * s
                + math.exp(self.cut_time - t)
                + math.exp(self.lower_bound - t)
                - 2.0 * math.exp(s - t)
                - self.cut_time - self.lower_bound
            )
        return pre_factor * integral

    def _psmc_prob(self, rho: float, s: float, t1: float, t2: float) -> float:
        """PSMC probability over interval [t1, t2] given source at s.

        Mirrors TSP_smc::psmc_prob.
        """
        l = 2.0 * s - self.lower_bound - self.cut_time
        if t1 == s == t2:
            base = math.exp(-rho * l)
        elif t1 < s < t2:
            base = math.exp(-rho * l)
        else:
            base = 0.0

        gap = 0.0
        if t2 > t1:
            uq = self._psmc_cdf(rho, s, t2)
            lq = self._psmc_cdf(rho, s, t1)
            gap = max(uq - lq, 0.0)

        prob = base + gap
        # clamp to [0, 1]
        return max(0.0, min(1.0, prob))

    def _get_prop(self, lb1: float, ub1: float, lb2: float, ub2: float) -> float:
        """Proportion of [lb2, ub2] occupied by [lb1, ub1] in exponential measure."""
        if ub2 - lb2 < 1e-6:
            return 1.0
        p1 = math.exp(-lb1) - math.exp(-ub1)
        p2 = math.exp(-lb2) - math.exp(-ub2)
        return p1 / p2 if p2 > 0 else 1.0

    # ------------------------------------------------------------------
    # Private: transition matrix computation
    # ------------------------------------------------------------------

    def _compute_diagonals(self, rho: float) -> None:
        """Compute stay-in-place probabilities.  Mirrors compute_diagonals."""
        if rho == self.prev_rho:
            return
        lb = self.curr_intervals[0].lb
        ub = self.curr_intervals[-1].ub
        for i, iv in enumerate(self.curr_intervals):
            base = self._psmc_prob(rho, iv.time, lb, ub)
            diag = self._psmc_prob(rho, iv.time, iv.lb, iv.ub)
            self.diagonals[i] = diag / base if base > 0 else 0.0

    def _compute_lower_diagonals(self, rho: float) -> None:
        """Compute lower off-diagonal (from below).  Mirrors compute_lower_diagonals."""
        if rho == self.prev_rho:
            return
        lb = max(self.cut_time, self.curr_intervals[0].lb)
        ub = self.curr_intervals[-1].ub
        self.lower_diagonals[self.dim - 1] = 0.0
        for i in range(self.dim - 1):
            t = self.curr_intervals[i + 1].time
            base = self._psmc_prob(rho, t, lb, ub)
            ld = self._psmc_prob(rho, t, self.curr_intervals[i].lb, self.curr_intervals[i].ub)
            self.lower_diagonals[i] = ld / base if base > 0 else 0.0

    def _compute_upper_diagonals(self, rho: float) -> None:
        """Compute upper off-diagonal (from above).  Mirrors compute_upper_diagonals."""
        if rho == self.prev_rho:
            return
        lb = max(self.cut_time, self.curr_intervals[0].lb)
        ub = self.curr_intervals[-1].ub
        self.upper_diagonals[0] = 0.0
        for i in range(1, self.dim):
            t = self.curr_intervals[i - 1].time
            base = self._psmc_prob(rho, t, lb, ub)
            ud = self._psmc_prob(rho, t, self.curr_intervals[i].lb, self.curr_intervals[i].ub)
            self.upper_diagonals[i] = ud / base if base > 0 else 0.0

    def _compute_lower_sums(self) -> None:
        """Cumulative sum for lower triangular part.  Mirrors compute_lower_sums."""
        self.lower_sums[0] = 0.0
        fp = self.forward_probs[self.curr_index]
        for i in range(1, self.dim):
            self.lower_sums[i] = (
                self.upper_diagonals[i] * fp[i - 1]
                + self.factors[i] * self.lower_sums[i - 1]
            )

    def _compute_upper_sums(self) -> None:
        """Partial sums for upper triangular part.  Mirrors compute_upper_sums."""
        fp = self.forward_probs[self.curr_index]
        # upper_sums[i] = sum(fp[i+1:])
        self.upper_sums[self.dim - 1] = 0.0
        for i in range(self.dim - 2, -1, -1):
            self.upper_sums[i] = fp[i + 1] + self.upper_sums[i + 1]

    # ------------------------------------------------------------------
    # Private: emission computation
    # ------------------------------------------------------------------

    def _compute_emissions(
        self, mut_set: Set[float], branch: Branch, node: "Node"
    ) -> None:
        """Compute diff counts for pre-computed emission.

        Mirrors TSP_smc::compute_emissions.
        """
        self.emissions = [0.0, 0.0, 0.0, 0.0]
        for x in mut_set:
            sl = branch.lower_node.get_state(x)
            su = branch.upper_node.get_state(x)
            s0 = node.get_state(x)
            sm = 1.0 if (sl + su + s0 > 1.5) else 0.0
            self.emissions[0] += abs(sm - sl)
            self.emissions[1] += abs(sm - su)
            self.emissions[2] += abs(sm - s0)
            self.emissions[3] += abs(sl - su)

    def _compute_null_emit_probs(self, theta: float, query_node: "Node") -> None:
        if theta == self.prev_theta and query_node is self.prev_node:
            return
        for i, iv in enumerate(self.curr_intervals):
            self.null_emit_probs[i] = self.eh.null_emit(
                self.curr_branch, iv.time, theta, query_node
            )

    def _compute_mut_emit_probs(
        self,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        query_node: "Node",
    ) -> None:
        self._compute_emissions(mut_set, self.curr_branch, query_node)
        for i, iv in enumerate(self.curr_intervals):
            self.mut_emit_probs[i] = self.eh.emit(
                self.curr_branch, iv.time, theta, bin_size, self.emissions, query_node
            )

    def _compute_trace_back_probs(
        self,
        rho: float,
        interval: Interval,
        intervals: List[Interval],
    ) -> None:
        """Compute traceback probability for each interval.

        Mirrors TSP_smc::compute_trace_back_probs.
        """
        if rho == self.prev_rho:
            return
        self.trace_back_probs = [0.0] * len(intervals)
        for i, iv in enumerate(intervals):
            p = self._psmc_prob(rho, iv.time, interval.lb, interval.ub)
            if iv.lb < iv.ub:
                p = max(_EPSILON, p)
            self.trace_back_probs[i] = p

    # ------------------------------------------------------------------
    # Private: constraint setting at recombinations
    # ------------------------------------------------------------------

    def _sanity_check(self, r: "Recombination") -> None:
        """Zero point-mass intervals on wrong branch at inserted_node time.

        Mirrors TSP_smc::sanity_check.
        """
        for i, iv in enumerate(self.curr_intervals):
            if (iv.lb == iv.ub
                    and iv.lb == r.inserted_node.time
                    and iv.branch is not r.target_branch):
                self.forward_probs[self.curr_index][i] = 0.0

    def _set_interval_constraint(self, r: "Recombination") -> None:
        """Zero/clamp previous probs at source→merging transition.

        Mirrors TSP_smc::set_interval_constraint.
        """
        intervals = self._get_state_space(self.curr_index - 1)
        for i, iv in enumerate(intervals):
            if iv.ub <= r.start_time:
                self.forward_probs[self.curr_index - 1][i] = 0.0
            else:
                iv.lb = max(r.start_time, iv.lb)
                iv.fill_time()

    def _set_point_constraint(self, r: "Recombination") -> None:
        """Collapse previous probs onto point mass at inserted_node.

        Mirrors TSP_smc::set_point_constraint.
        """
        point_iv = self._search_point_interval(r)
        intervals = self._get_state_space(self.curr_index - 1)
        for i, iv in enumerate(intervals):
            if iv is point_iv:
                self.forward_probs[self.curr_index - 1][i] = 1.0
                iv.node = r.inserted_node
            else:
                self.forward_probs[self.curr_index - 1][i] = 0.0

    def _search_point_interval(self, r: "Recombination") -> Optional[Interval]:
        """Find the interval that should receive the point mass.

        Mirrors TSP_smc::search_point_interval.
        """
        t = r.inserted_node.time
        point_iv: Optional[Interval] = None

        # Priority 1: proper containing interval
        for iv in self.curr_intervals:
            if iv.lb < t < iv.ub:
                point_iv = iv
        # Priority 2: exact point interval
        for iv in self.curr_intervals:
            if iv.lb == iv.ub == t:
                point_iv = iv

        if point_iv is not None:
            return point_iv

        # Fallback: two candidate intervals straddle t; pick one not coming
        # from inserted_node's branch via source_interval chain.
        candidates = [iv for iv in self.curr_intervals if iv.lb <= t <= iv.ub]
        if len(candidates) < 2:
            raise RuntimeError(f"TSP transfer_sample: expected 2 candidate intervals, got {len(candidates)}")

        test_iv = candidates[0]
        while id(test_iv) in self.source_interval:
            test_iv = self.source_interval[id(test_iv)]
            if (test_iv.branch.upper_node is r.inserted_node
                    or test_iv.branch.lower_node is r.inserted_node):
                return candidates[1]
        return candidates[0]

    # ------------------------------------------------------------------
    # Private: state-space lookup
    # ------------------------------------------------------------------

    def _get_state_space(self, x: int) -> List[Interval]:
        """Return the state space valid at step x (floor lookup)."""
        # Find the largest key <= x
        keys = sorted(k for k in self.state_spaces.keys() if k <= x)
        if not keys:
            return self.curr_intervals
        return self.state_spaces[keys[-1]]

    def _get_prev_breakpoint(self, x: int) -> int:
        """Return the largest state-space key <= x."""
        keys = sorted(k for k in self.state_spaces.keys() if k <= x)
        if not keys:
            return 0
        return keys[-1]

    def _get_interval_index(
        self, interval: Interval, intervals: List[Interval]
    ) -> int:
        for i, iv in enumerate(intervals):
            if iv is interval:
                return i
        return 0

    # ------------------------------------------------------------------
    # Private: traceback sampling
    # ------------------------------------------------------------------

    def _sample_curr_interval(self, x: int) -> Interval:
        intervals = self._get_state_space(x)
        # If the state space at x is empty (degenerate branch), search backwards
        if not intervals:
            for k in sorted(self.state_spaces.keys(), reverse=True):
                if self.state_spaces[k]:
                    intervals = self.state_spaces[k]
                    x = k
                    break
        fp = self.forward_probs[x]
        ws = sum(fp)
        q = self._random()
        w = ws * q
        for i, iv in enumerate(intervals):
            w -= fp[i]
            if w <= 0:
                self.sample_index = i
                return iv
        self.sample_index = len(intervals) - 1
        return intervals[-1]

    def _sample_prev_interval(self, interval: Interval, x: int) -> Interval:
        intervals = self._get_state_space(x)
        if not intervals:
            return interval
        self.lower_bound = intervals[0].lb
        rho = self.rhos[x]
        self._compute_trace_back_probs(rho, interval, intervals)
        fp = self.forward_probs[x]
        ws = sum(
            self.trace_back_probs[i] * fp[i]
            for i in range(len(intervals))
            if intervals[i] is not interval
        )
        if ws <= 0:
            # Fallback: return the interval whose time is closest to the
            # current interval's representative time.  This preserves
            # temporal continuity and avoids systematic bias.
            target_t = interval.time  # exponential median set by fill_time()
            best_i = min(range(len(intervals)), key=lambda i: abs(intervals[i].time - target_t))
            self.sample_index = best_i
            return intervals[best_i]
        q = self._random()
        w = ws * q
        for i, iv in enumerate(intervals):
            if iv is not interval:
                w -= self.trace_back_probs[i] * fp[i]
                if w <= 0:
                    self.sample_index = i
                    return iv
        self.sample_index = len(intervals) - 1
        return intervals[-1]

    def _sample_source_interval(self, interval: Interval, x: int) -> Interval:
        src = self.source_interval[id(interval)]
        intervals = self._get_state_space(x)
        self.sample_index = self._get_interval_index(src, intervals)
        return src

    def _sample_recomb_interval(self, interval: Interval, x: int) -> Interval:
        if interval.lb == interval.ub:
            # Point mass: fall back to plain sampling
            return self._sample_curr_interval(x)
        intervals = self._get_state_space(x)
        fp = self.forward_probs[x]
        ws = sum(
            self._recomb_prob(iv.time, interval.lb, interval.ub) * fp[i]
            for i, iv in enumerate(intervals)
        )
        if ws <= 0:
            return self._sample_curr_interval(x)
        q = self._random()
        w = ws * q
        for i, iv in enumerate(intervals):
            w -= self._recomb_prob(iv.time, interval.lb, interval.ub) * fp[i]
            if w <= 0:
                self.sample_index = i
                return iv
        self.sample_index = len(intervals) - 1
        return intervals[-1]

    def _trace_back_helper(self, interval: Interval, x: int) -> int:
        """Walk backward from step x.  Mirrors TSP_smc::trace_back_helper."""
        y = self._get_prev_breakpoint(x)
        q = self._random()
        p = 1.0
        intervals = self._get_state_space(x)
        if not intervals:
            return y  # degenerate: no state, skip to previous breakpoint
        self.lower_bound = intervals[0].lb
        self.trace_back_probs = [0.0] * len(intervals)

        while p > q and x > y:
            rho = self.rhos[x - 1]
            self._compute_trace_back_probs(rho, interval, intervals)
            self.prev_rho = rho
            prev_fp = self.forward_probs[x - 1]
            all_prob = sum(
                self.trace_back_probs[i] * prev_fp[i]
                for i in range(len(intervals))
            )
            if all_prob <= 0:
                raise RuntimeError("TSP trace_back_helper: zero all_prob")
            non_recomb = self.trace_back_probs[self.sample_index] * prev_fp[self.sample_index]
            shrinkage = non_recomb / all_prob
            p *= shrinkage
            if p <= q:
                return x
            x -= 1

        return y

    def _sample_joining_node(self, interval: Interval) -> Node:
        """Sample a coalescence node from *interval*.

        Mirrors TSP_smc::sample_joining_node (single-arg version).
        """
        global _counter
        if interval.node is not None:
            return interval.node
        t = self._exp_median(interval.lb, interval.ub)
        n = _new_node(t)
        n.index = _counter
        _counter += 1
        return n

    # ------------------------------------------------------------------
    # Private: time sampling helpers
    # ------------------------------------------------------------------

    def _exp_median(self, lb: float, ub: float) -> float:
        """Sample a time uniformly in exponential quantile space.

        Mirrors TSP_smc::exp_median.
        """
        if math.isinf(ub):
            return lb + 2.0 * self._random()
        if ub - lb <= 0.005:
            return (0.45 + 0.1 * self._random()) * (ub - lb) + lb
        if lb > 10:
            return (0.45 + 0.1 * self._random()) * (ub - lb) + lb
        lq = 1.0 - math.exp(-lb)
        uq = 1.0 - math.exp(-ub)
        mq = (0.45 + 0.1 * self._random()) * (uq - lq) + lq
        m = -math.log(1.0 - mq)
        return max(lb, min(ub, m))
