"""
Interval and IntervalInfo — time-height intervals on ARG branches.

Mirrors Interval.cpp / Interval.hpp.

An *Interval* represents a window [lb, ub] in coalescent time on a specific
Branch.  The BSP stores a list of current Intervals as its forward-pass
state space; the TSP does the same.

*IntervalInfo* is a lightweight, hashable summary used as a dict key
inside BSP.transfer() to accumulate transfer weights from multiple source
intervals into a single target interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .branch import Branch
    from .node import Node


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


class Interval:
    """A (branch, lb, ub) time interval with associated forward-HMM state.

    Attributes
    ----------
    branch:      The ARG branch this interval lives on.
    lb:          Lower time bound.
    ub:          Upper time bound.
    start_pos:   Index into the coordinate grid where this interval was
                 created (used during traceback).
    weight:      Coalescent probability mass in [lb, ub] (set by BSP).
    time:        Representative time point in [lb, ub] (median or median-CDF).
    source_weights:   Weights of source intervals (BSP transfer).
    source_intervals: Corresponding source Interval objects.
    node:        Optional Node pointer (used by TSP for point-mass intervals).
    """

    __slots__ = (
        "branch", "lb", "ub", "start_pos",
        "weight", "time",
        "source_weights", "source_intervals",
        "node",
    )

    def __init__(
        self,
        branch: "Branch",
        lb: float,
        ub: float,
        start_pos: int,
    ) -> None:
        assert lb <= ub, f"lb={lb} > ub={ub}"
        self.branch = branch
        self.lb = lb
        self.ub = ub
        self.start_pos = start_pos
        self.weight: float = 0.0
        self.time: float = 0.0
        self.source_weights: List[float] = []
        self.source_intervals: List["Interval"] = []
        self.node: Optional["Node"] = None

    # ------------------------------------------------------------------
    # Convenience methods matching C++ API
    # ------------------------------------------------------------------

    def assign_weight(self, w: float) -> None:
        self.weight = w

    def assign_time(self, t: float) -> None:
        assert self.lb <= t <= self.ub, f"time {t} outside [{self.lb}, {self.ub}]"
        self.time = t

    def fill_time(self) -> None:
        """Set self.time to the exponential-median of [lb, ub].

        Mirrors Interval::fill_time() in the C++ code.
        """
        lb, ub = self.lb, self.ub
        if math.isinf(ub):
            self.time = lb + math.log(2)
            return
        if abs(lb - ub) < 1e-3:
            self.time = 0.5 * (lb + ub)
            return
        lq = 1.0 - math.exp(-lb)
        uq = 1.0 - math.exp(-ub)
        if uq - lq < 1e-3:
            self.time = 0.5 * (lb + ub)
        else:
            q = 0.5 * (lq + uq)
            self.time = -math.log(1.0 - q)
        # Clamp to avoid floating-point drift
        self.time = max(lb, min(ub, self.time))

    def full(self, cut_time: float) -> bool:
        """True iff this interval spans the entire branch above cut_time."""
        lb_expected = max(cut_time, self.branch.lower_node.time)
        ub_expected = self.branch.upper_node.time
        return self.lb == lb_expected and self.ub == ub_expected

    # ------------------------------------------------------------------
    # Ordering (used for sorting state spaces and as set members)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return False
        return (
            self.start_pos == other.start_pos
            and self.branch == other.branch
            and self.ub == other.ub
            and self.lb == other.lb
        )

    def __hash__(self) -> int:
        return hash((self.start_pos, hash(self.branch), self.ub, self.lb))

    def __lt__(self, other: "Interval") -> bool:
        if self.start_pos != other.start_pos:
            return self.start_pos < other.start_pos
        if self.branch != other.branch:
            return self.branch < other.branch
        if self.ub != other.ub:
            return self.ub < other.ub
        return self.lb < other.lb

    def __repr__(self) -> str:
        return (
            f"Interval(branch={self.branch}, "
            f"[{self.lb:.3g}, {self.ub:.3g}], start={self.start_pos})"
        )


# ---------------------------------------------------------------------------
# IntervalInfo  (Interval_info in C++)
# ---------------------------------------------------------------------------


class IntervalInfo:
    """Lightweight key for BSP.transfer() accumulation maps.

    Mirrors C++ Interval_info — a (branch, lb, ub) triple with a seed_pos
    tiebreaker.  Must be hashable so it can serve as a dict key.
    """

    __slots__ = ("branch", "lb", "ub", "seed_pos")

    def __init__(
        self,
        branch: "Branch",
        lb: float,
        ub: float,
        seed_pos: int = 0,
    ) -> None:
        assert lb <= ub
        self.branch = branch
        self.lb = lb
        self.ub = ub
        self.seed_pos = seed_pos

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalInfo):
            return False
        return (
            self.seed_pos == other.seed_pos
            and self.branch == other.branch
            and self.ub == other.ub
            and self.lb == other.lb
        )

    def __hash__(self) -> int:
        return hash((self.seed_pos, hash(self.branch), self.ub, self.lb))

    def __lt__(self, other: "IntervalInfo") -> bool:
        if self.seed_pos != other.seed_pos:
            return self.seed_pos < other.seed_pos
        if self.branch != other.branch:
            return self.branch < other.branch
        if self.ub != other.ub:
            return self.ub < other.ub
        return self.lb < other.lb

    def __repr__(self) -> str:
        return (
            f"IntervalInfo(branch={self.branch}, "
            f"[{self.lb:.3g}, {self.ub:.3g}], seed={self.seed_pos})"
        )
