"""
CoalescentCalculator — piecewise-exponential coalescent CDF/quantile.

Mirrors Coalescent_calculator.cpp / Coalescent_calculator.hpp.

Given a set of branches spanning a range of times, computes the
probability that a new lineage coalesces in any time interval [lb, ub].
The coalescent rate at time t is the number of lineages currently alive
at t (i.e., spanning t).

The CDF is piecewise-exponential: within each interval the rate is
constant, so the survival function is exp(−rate * Δt).
"""
from __future__ import annotations

import bisect
import math
from typing import List, Set, Tuple, TYPE_CHECKING

from sortedcontainers import SortedDict

if TYPE_CHECKING:
    from ..data.branch import Branch


class CoalescentCalculator:
    """Piecewise-exponential coalescent CDF for a set of branches.

    Usage::

        cc = CoalescentCalculator(cut_time=0.0)
        cc.compute(set_of_branches)
        p = cc.weight(lb, ub)   # probability of coalescence in [lb, ub]
        t = cc.time(lb, ub)     # representative time in [lb, ub]
    """

    def __init__(self, cut_time: float) -> None:
        self.cut_time = cut_time

        # rate_changes[time] = Δrate (positive when branches start, negative when they end)
        self._rate_changes: SortedDict = SortedDict()
        # rates[time] = current coalescent rate (cumulative sum of rate_changes)
        self._rates: SortedDict = SortedDict()
        # Cumulative probabilities indexed by time (SortedDict[time → cum_prob])
        self._cum_probs: SortedDict = SortedDict()
        # Same data as parallel lists for quantile lookup (sorted by cum_prob)
        self._prob_vals: List[float] = []   # cum_prob values (sorted)
        self._prob_times: List[float] = []  # corresponding times

        self.min_time: float = 0.0
        self.max_time: float = math.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, branches: Set["Branch"]) -> None:
        """Recompute the CDF from *branches*.

        Mirrors Coalescent_calculator::compute.
        """
        self._compute_rate_changes(branches)
        self._compute_rates()
        self._compute_probs_quantiles()

    def weight(self, lb: float, ub: float) -> float:
        """Return the probability of coalescence in [lb, ub].

        Mirrors Coalescent_calculator::weight.
        """
        p = self.prob(ub) - self.prob(lb)
        return p

    def time(self, lb: float, ub: float) -> float:
        """Return a representative coalescence time in [lb, ub].

        Uses the exponential median; falls back to the midpoint when the
        interval is tiny.  Mirrors Coalescent_calculator::time.
        """
        if math.isinf(ub):
            return lb + math.log(2)
        if ub - lb < 1e-3:
            return 0.5 * (lb + ub)
        lq = self.prob(lb)
        uq = self.prob(ub)
        if uq - lq < 1e-3:
            return 0.5 * (lb + ub)
        mid = 0.5 * (lq + uq)
        t = self.quantile(mid)
        return max(lb, min(ub, t))

    def prob(self, x: float) -> float:
        """Return cumulative coalescence probability at time *x*.

        Mirrors Coalescent_calculator::prob.
        """
        if not self._cum_probs:
            return 0.0
        if x >= self.max_time:
            x = self.max_time
        elif x <= self.min_time:
            return 0.0

        # Exact hit
        if x in self._cum_probs:
            return self._cum_probs[x]

        # Interpolate in the piecewise-exponential CDF
        u_idx = self._cum_probs.bisect_right(x)
        l_idx = u_idx - 1

        if l_idx < 0:
            return 0.0
        if u_idx >= len(self._cum_probs):
            return self._cum_probs[self._cum_probs.keys()[-1]]

        l_key = self._cum_probs.keys()[l_idx]
        u_key = self._cum_probs.keys()[u_idx]

        base_prob = self._cum_probs[l_key]
        rate = self._rates.get(l_key, 0)

        if rate == 0:
            return base_prob

        delta_t = u_key - l_key
        delta_p = self._cum_probs[u_key] - base_prob
        new_delta_t = x - l_key

        # Interpolation formula from the C++ code:
        #   new_delta_p = delta_p * expm1(-rate * new_delta_t) / expm1(-rate * delta_t)
        denom = math.expm1(-rate * delta_t)
        if abs(denom) < 1e-15:
            new_delta_p = delta_p * new_delta_t / delta_t
        else:
            new_delta_p = delta_p * math.expm1(-rate * new_delta_t) / denom

        return base_prob + new_delta_p

    def quantile(self, p: float) -> float:
        """Return the time t such that prob(t) == p.

        Mirrors Coalescent_calculator::quantile.
        """
        if not self._prob_vals:
            return self.min_time

        # Find the interval [l, u] where the cum_prob crosses p
        idx = bisect.bisect_right(self._prob_vals, p)
        l_idx = idx - 1
        u_idx = idx

        if l_idx < 0:
            l_idx = 0
        if u_idx >= len(self._prob_vals):
            u_idx = len(self._prob_vals) - 1

        l_time = self._prob_times[l_idx]
        u_time = self._prob_times[u_idx]
        l_prob = self._prob_vals[l_idx]
        u_prob = self._prob_vals[u_idx]

        base_time = l_time
        rate = self._rates.get(l_time, 0)
        delta_t = u_time - l_time
        delta_p = u_prob - l_prob

        if delta_p < 1e-15:
            return base_time

        new_delta_p = p - l_prob

        # Inverse formula from C++:
        #   new_delta_t = -log(1 - new_delta_p/delta_p * (1 - exp(-rate*delta_t))) / rate
        if rate == 0:
            new_delta_t = delta_t * new_delta_p / delta_p
        else:
            frac = new_delta_p / delta_p * (1.0 - math.exp(-rate * delta_t))
            arg = 1.0 - frac
            if arg <= 0:
                return u_time
            new_delta_t = -math.log(arg) / rate

        return base_time + new_delta_t

    # ------------------------------------------------------------------
    # Private: CDF construction
    # ------------------------------------------------------------------

    def _compute_rate_changes(self, branches: Set["Branch"]) -> None:
        """Record +1 at branch start and -1 at branch end.

        Mirrors C++ exactly: rate_changes[ub] -= 1 even for ub=inf,
        so max_time = inf for any branch reaching the root sentinel.
        """
        self._rate_changes = SortedDict()
        for b in branches:
            lb = max(self.cut_time, b.lower_node.time)
            ub = b.upper_node.time
            self._rate_changes[lb] = self._rate_changes.get(lb, 0) + 1
            self._rate_changes[ub] = self._rate_changes.get(ub, 0) - 1

        if not self._rate_changes:
            return
        self.min_time = self._rate_changes.keys()[0]
        self.max_time = self._rate_changes.keys()[-1]

    def _compute_rates(self) -> None:
        """Cumulative sum of rate_changes → piecewise constant rate."""
        self._rates = SortedDict()
        curr = 0
        for t, delta in self._rate_changes.items():
            curr += delta
            self._rates[t] = curr

    def _compute_probs_quantiles(self) -> None:
        """Build the piecewise CDF from the rates.

        Mirrors Coalescent_calculator::compute_probs_quantiles.
        """
        self._cum_probs = SortedDict()
        if not self._rates:
            return

        rate_keys = list(self._rates.keys())
        prev_prob = 1.0
        cum_prob = 0.0

        for i in range(len(rate_keys) - 1):
            curr_rate = self._rates[rate_keys[i]]
            prev_time = rate_keys[i]
            next_time = rate_keys[i + 1]

            if curr_rate > 0:
                next_prob = prev_prob * math.exp(-curr_rate * (next_time - prev_time))
                cum_prob += prev_prob - next_prob
            else:
                next_prob = prev_prob

            self._cum_probs[next_time] = cum_prob
            prev_prob = next_prob

        # Sentinel at min_time with cumulative probability 0
        self._cum_probs[self.min_time] = 0.0

        # Build parallel arrays sorted by cum_prob for quantile lookup.
        # (Mirrors C++ quantiles set sorted by {cum_prob, time}.)
        pairs: List[Tuple[float, float]] = sorted(
            (cp, t) for t, cp in self._cum_probs.items()
        )
        self._prob_vals = [cp for cp, t in pairs]
        self._prob_times = [t for cp, t in pairs]
