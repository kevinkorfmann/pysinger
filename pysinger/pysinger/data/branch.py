"""
Branch — a directed edge between two nodes, hashable by node identity.

Mirrors Branch.cpp / Branch.hpp.

In C++ each Branch holds shared_ptr<Node> lower_node and upper_node and
compares them by pointer identity.  Here we do the same using Python
object identity (id()).  The class is immutable after construction so it
can safely serve as a dict key or set member.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


class Branch:
    """Directed edge (lower_node, upper_node) in the ARG.

    Invariant: lower_node.time <= upper_node.time (except for the null
    branch where both are None).

    Hashability: based on Python object identity of the two node pointers,
    exactly as the C++ `branch_hash` struct.
    """

    __slots__ = ("lower_node", "upper_node")

    def __init__(
        self,
        lower_node: Optional["Node"] = None,
        upper_node: Optional["Node"] = None,
    ) -> None:
        object.__setattr__(self, "lower_node", lower_node)
        object.__setattr__(self, "upper_node", upper_node)

    # Python immutability (no frozen dataclass because Node is unhashable
    # as a dataclass — we need identity-based equality for Branch).
    def __setattr__(self, name, value):
        raise AttributeError("Branch is immutable")

    # ------------------------------------------------------------------
    # Comparison & hashing (mirrors C++ operator== / branch_hash)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Branch):
            return False
        return (
            self.lower_node is other.lower_node
            and self.upper_node is other.upper_node
        )

    def __hash__(self) -> int:
        return hash((id(self.lower_node), id(self.upper_node)))

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: "Branch") -> bool:
        """Ordering for sets/sorted containers.  Mirrors C++ operator<."""
        def _key(n):
            # None sorts before any real node (time=inf for root placeholder)
            if n is None:
                return (float("inf"), -1)
            return (n.time, n.index)

        su, ou = _key(self.upper_node), _key(other.upper_node)
        if su != ou:
            return su < ou
        return _key(self.lower_node) < _key(other.lower_node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Branch length in time units.  Returns inf for branches to root."""
        if self.lower_node is None or self.upper_node is None:
            return float("inf")
        return self.upper_node.time - self.lower_node.time

    def is_null(self) -> bool:
        """True iff this is the null (empty) branch."""
        return self.lower_node is None and self.upper_node is None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        def _fmt(n):
            return "None" if n is None else f"N{n.index}(t={n.time:.3g})"

        return f"Branch({_fmt(self.lower_node)}, {_fmt(self.upper_node)})"

    def __bool__(self) -> bool:
        return not self.is_null()
