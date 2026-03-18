"""
Node — a vertex in the ARG carrying allele state information.

Mirrors Node.cpp / Node.hpp.  The C++ version uses a std::map with a
cached iterator; here we use sortedcontainers.SortedDict and rely on its
O(log n) bisect operations, which are fast enough for a readable PoC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from sortedcontainers import SortedDict


@dataclass
class Node:
    """A node in the Ancestral Recombination Graph.

    Parameters
    ----------
    time:
        Coalescence time (in units of 2Ne generations).  Leaf nodes have
        time == 0; the global root sentinel has time == inf.
    index:
        Integer label.  Sample nodes receive 0-based indices; internal nodes
        receive larger indices; the root sentinel has index == -1.
    mutation_sites:
        Maps genomic position → allele state (1 = derived).  A sentinel
        entry at position -1 with state 0 is kept so look-ups before the
        first real position always succeed.
    """

    time: float
    index: int = 0
    # The sentinel at -1 mirrors the C++ initialiser `mutation_sites[-1] = 0`.
    mutation_sites: SortedDict = field(
        default_factory=lambda: SortedDict({-1: 0})
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_mutation(self, pos: float) -> None:
        """Record that this node carries the derived allele at *pos*."""
        self.mutation_sites[pos] = 1

    def get_state(self, pos: float) -> float:
        """Return the allele state at *pos* (exact match only).

        Mirrors C++ Node::get_state: returns the stored value if there is an
        exact key at *pos*, otherwise 0.  This means mutations are point
        events — a mutation at position 100 does NOT carry its state forward
        to position 101 or beyond.
        """
        return self.mutation_sites.get(pos, 0)

    def write_state(self, pos: float, state: float) -> None:
        """Overwrite the allele state; state==0 removes the entry."""
        if state == 0:
            # Never remove the sentinel at -1.
            if pos != -1:
                self.mutation_sites.pop(pos, None)
        elif state == 1:
            self.mutation_sites[pos] = state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Node(time={self.time:.4g}, index={self.index})"

    def __hash__(self):
        # Identity-based hash so Node objects can live in sets / be dict keys.
        return id(self)

    def __eq__(self, other):
        return self is other

    def __lt__(self, other):
        # Ordering used by Tree.sample_cut_point and similar helpers.
        if self.time != other.time:
            return self.time < other.time
        return self.index < other.index
