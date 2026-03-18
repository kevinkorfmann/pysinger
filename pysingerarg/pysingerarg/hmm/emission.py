"""
Emission models for the BSP/TSP forward HMM.

Mirrors Binary_emission.cpp, Polar_emission.cpp, and Emission.hpp.

The abstract base class Emission defines two methods:
  null_emit(branch, time, theta, query_node)   — no mutation in bin
  mut_emit(branch, time, theta, bin_size, mut_set, query_node) — mutations in bin

BinaryEmission — symmetric infinite-sites model (C++ Binary_emission).
PolarEmission  — polarised (ancestral/derived) model (C++ Polar_emission).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.branch import Branch
    from ..data.node import Node


class Emission(ABC):
    """Abstract emission model for the HMM."""

    @abstractmethod
    def null_emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        node: "Node",
    ) -> float:
        """Return the emission probability when no mutations fall in the bin."""
        ...

    @abstractmethod
    def mut_emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        node: "Node",
    ) -> float:
        """Return the emission probability when mut_set mutations fall in the bin."""
        ...

    def emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        bin_size: float,
        emissions: list,
        node: "Node",
    ) -> float:
        """Emit using pre-computed diff counts (used by TSP)."""
        ...


# ---------------------------------------------------------------------------
# BinaryEmission
# ---------------------------------------------------------------------------


class BinaryEmission(Emission):
    """Symmetric infinite-sites emission model.

    Mirrors Binary_emission.cpp.

    The emission probability for a branch (lower, upper) and query node with
    representative time *t* on the branch is:
        P(data) ∝ exp(-θ * l_lower) * (θ/bin_size)^s_lower
                 * exp(-θ * l_upper) * (θ/bin_size)^s_upper
                 * exp(-θ * l_query) * (θ/bin_size)^s_query

    divided by the "old" probability (without the threaded lineage) to get
    the ratio used in the HMM.
    """

    def _calculate_prob(self, theta: float, bin_size: float, s: int) -> float:
        """P(s mutations | theta, bin_size) = exp(-theta) * (theta/bin_size)^s."""
        if math.isinf(theta):
            return 1.0
        unit_theta = theta / bin_size
        return math.exp(-theta) * (unit_theta ** s)

    def _get_diff(
        self,
        mut_set: Set[float],
        branch: "Branch",
        node: "Node",
    ) -> list:
        """Compute (s_lower, s_upper, s_query, s_old) difference counts."""
        d = [0, 0, 0, 0]
        for x in mut_set:
            sl = branch.lower_node.get_state(x)
            su = branch.upper_node.get_state(x)
            s0 = node.get_state(x)
            sm = 1 if (sl + su + s0 > 1.5) else 0
            d[0] += abs(sm - sl)
            d[1] += abs(sm - su)
            d[2] += abs(sm - s0)
            d[3] += abs(sl - su)
        return d

    def null_emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        node: "Node",
    ) -> float:
        ll = time - branch.lower_node.time
        lu = branch.upper_node.time - time
        l0 = time - node.time
        emit_prob = (
            self._calculate_prob(ll * theta, 1, 0)
            * (self._calculate_prob(lu * theta, 1, 0) if not math.isinf(lu) else 1.0)
            * self._calculate_prob(l0 * theta, 1, 0)
        )
        if not math.isinf(lu):
            old_prob = self._calculate_prob((ll + lu) * theta, 1, 0)
        else:
            old_prob = 1.0
        return emit_prob / old_prob if old_prob > 0 else 1.0

    def mut_emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        node: "Node",
    ) -> float:
        ll = time - branch.lower_node.time
        lu = branch.upper_node.time - time
        l0 = time - node.time
        diff = self._get_diff(mut_set, branch, node)
        emit_prob = (
            self._calculate_prob(ll * theta, bin_size, diff[0])
            * self._calculate_prob(lu * theta, bin_size, diff[1])
            * self._calculate_prob(l0 * theta, bin_size, diff[2])
        )
        old_prob = self._calculate_prob((ll + lu) * theta, bin_size, diff[3])
        if old_prob <= 0:
            return 1e-20
        return max(emit_prob / old_prob, 1e-20)

    def emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        bin_size: float,
        emissions: list,
        node: "Node",
    ) -> float:
        ll = time - branch.lower_node.time
        lu = branch.upper_node.time - time
        l0 = time - node.time
        emit_prob = (
            self._calculate_prob(ll * theta, bin_size, emissions[0])
            * self._calculate_prob(lu * theta, bin_size, emissions[1])
            * self._calculate_prob(l0 * theta, bin_size, emissions[2])
        )
        old_prob = self._calculate_prob((ll + lu) * theta, bin_size, emissions[3])
        if old_prob <= 0:
            return 1.0
        return emit_prob / old_prob


# ---------------------------------------------------------------------------
# PolarEmission
# ---------------------------------------------------------------------------


class PolarEmission(Emission):
    """Polarised emission model for ancestral/derived alleles.

    Mirrors Polar_emission.cpp.

    Additional parameters:
        penalty:       Cost of derived alleles in the query node.
        ancestral_prob: Prior probability that allele is ancestral at root.
    """

    def __init__(self, penalty: float = 1.0, ancestral_prob: float = 0.5) -> None:
        self.penalty = penalty
        self.ancestral_prob = ancestral_prob
        self._root_reward: float = 1.0

    def _null_prob(self, theta: float) -> float:
        if math.isinf(theta):
            return 1.0
        return math.exp(-theta)

    def _mut_prob_single(self, theta: float, bin_size: float, s: float) -> float:
        """Prob of *s* mutations on a branch of length theta."""
        if math.isinf(theta):
            return 1.0
        unit_theta = theta / bin_size
        return unit_theta ** abs(s)

    def _get_diff(
        self,
        m: float,
        branch: "Branch",
        node: "Node",
    ) -> list:
        """Compute directional difference vector."""
        sl = branch.lower_node.get_state(m)
        su = branch.upper_node.get_state(m)
        s0 = node.get_state(m)
        sm = 1 if (sl + su + s0 > 1.5) else 0

        # Root reward
        if branch.upper_node.index == -1:
            if sm == 0 and sl == 1:
                self._root_reward = self.ancestral_prob / (1.0 - self.ancestral_prob)
            else:
                self._root_reward = 1.0
        else:
            self._root_reward = 1.0

        return [sl - sm, sm - su, s0 - sm, sl - su]

    def null_emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        node: "Node",
    ) -> float:
        ll = time - branch.lower_node.time
        lu = branch.upper_node.time - time
        l0 = time - node.time
        if not math.isinf(lu):
            return self._null_prob(theta * l0)
        else:
            return self._null_prob(theta * (ll + l0))

    def mut_emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        bin_size: float,
        mut_set: Set[float],
        node: "Node",
    ) -> float:
        ll = time - branch.lower_node.time
        lu = branch.upper_node.time - time
        l0 = time - node.time
        emit_prob = 1.0
        old_prob = 1.0
        self._root_reward = 1.0
        for m in mut_set:
            diff = self._get_diff(m, branch, node)
            emit_prob *= (
                self._mut_prob_single(ll * theta, bin_size, diff[0])
                * self._mut_prob_single(lu * theta, bin_size, diff[1])
                * self._mut_prob_single(l0 * theta, bin_size, diff[2])
            )
            if diff[2] >= 1:
                emit_prob *= self.penalty
            old_prob *= self._mut_prob_single((ll + lu) * theta, bin_size, diff[3])

        if not math.isinf(lu):
            emit_prob *= self._null_prob(theta * l0)
        else:
            emit_prob *= self._null_prob(theta * (l0 + ll))

        if old_prob <= 0:
            return 1e-20
        return max(emit_prob / old_prob * self._root_reward, 1e-20)

    def emit(
        self,
        branch: "Branch",
        time: float,
        theta: float,
        bin_size: float,
        emissions: list,
        node: "Node",
    ) -> float:
        ll = time - branch.lower_node.time
        lu = branch.upper_node.time - time
        emit_prob = (
            self._mut_prob_single(ll * theta, bin_size, emissions[0])
            * self._mut_prob_single(lu * theta, bin_size, emissions[1])
            * self._mut_prob_single((time - node.time) * theta, bin_size, emissions[2])
        )
        emit_prob *= self._null_prob(theta * (time - node.time))
        old_prob = (
            self._mut_prob_single((ll + lu) * theta, bin_size, emissions[3])
            * self._null_prob((ll + lu) * theta)
        )
        if old_prob <= 0:
            return 1.0
        return emit_prob / old_prob
