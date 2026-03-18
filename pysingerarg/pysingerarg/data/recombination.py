"""
Recombination — the ARG topology change at a single recombination breakpoint.

Mirrors Recombination.cpp / Recombination.hpp.

A Recombination stores the set of branches deleted and inserted at a
genomic position `pos`.  From the deleted/inserted sets it derives the
key named branches (source, target, merging, recombined, transfer branches)
and the associated times (start_time).  These are used by the BSP/TSP
transfer steps and by the MCMC threader.
"""
from __future__ import annotations

import sys
from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .branch import Branch
    from .node import Node


# We import Branch lazily inside methods where needed to avoid circular
# imports at module load time.


class Recombination:
    """Topology change record for one recombination breakpoint.

    Attributes
    ----------
    pos:              Genomic position of the breakpoint (set externally).
    deleted_branches: Branches that exist *before* (left of) `pos`.
    inserted_branches:Branches that exist *after* (right of) `pos`.

    Derived attributes (computed by find_recomb_info):
    source_branch:    The lineage that recombines.
    target_branch:    The lineage it joins at coalescence.
    inserted_node:    New internal node created at the coalescence.
    deleted_node:     Old internal node removed by the topology change.
    merging_branch:   The branch that takes over after the removed node.
    recombined_branch:Part of the source lineage below start_time.
    start_time:       Height at which recombination begins.
    lower/upper_transfer_branch: For BSP interval transfer.
    source_sister_branch, source_parent_branch: auxiliary.
    """

    def __init__(
        self,
        deleted_branches: Optional[Set] = None,
        inserted_branches: Optional[Set] = None,
    ) -> None:
        from .branch import Branch

        self.pos: float = 0.0

        self.deleted_branches: Set[Branch] = (
            set(deleted_branches) if deleted_branches else set()
        )
        self.inserted_branches: Set[Branch] = (
            set(inserted_branches) if inserted_branches else set()
        )

        # Derived by find_nodes() / find_recomb_info()
        self.deleted_node: Optional["Node"] = None
        self.inserted_node: Optional["Node"] = None
        self.source_branch: "Branch" = Branch()
        self.target_branch: "Branch" = Branch()
        self.merging_branch: "Branch" = Branch()
        self.recombined_branch: "Branch" = Branch()
        self.source_sister_branch: "Branch" = Branch()
        self.source_parent_branch: "Branch" = Branch()
        self.lower_transfer_branch: "Branch" = Branch()
        self.upper_transfer_branch: "Branch" = Branch()
        self.start_time: float = 0.0

        if self.deleted_branches or self.inserted_branches:
            self._simplify_branches()
            self._find_nodes()

    def set_pos(self, x: float) -> None:
        self.pos = x

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def affect(self, b: "Branch") -> bool:
        """True iff *b* is in deleted_branches."""
        return b in self.deleted_branches

    def create(self, b: "Branch") -> bool:
        """True iff *b* is in inserted_branches."""
        return b in self.inserted_branches

    # ------------------------------------------------------------------
    # Branch-tracing (used by ARG.remove / ARG.add)
    # ------------------------------------------------------------------

    def trace_forward(self, t: float, curr_branch: "Branch") -> "Branch":
        """Return the branch at time *t* that *curr_branch* maps to after pos.

        Mirrors Recombination::trace_forward.
        """
        from .branch import Branch

        _NULL = Branch()
        _INT_MAX = sys.maxsize

        if self.pos == 0 or self.pos == _INT_MAX:
            return _NULL
        if not self.affect(curr_branch):
            return curr_branch

        if curr_branch == self.source_branch:
            if t >= self.start_time:
                return _NULL
            else:
                return self.recombined_branch
        elif curr_branch == self.target_branch:
            if t > self.inserted_node.time:
                return self.upper_transfer_branch
            else:
                return self.lower_transfer_branch
        else:
            return self.merging_branch

    def trace_backward(self, t: float, curr_branch: "Branch") -> "Branch":
        """Return the branch that maps to *curr_branch* before pos.

        Mirrors Recombination::trace_backward.
        """
        from .branch import Branch

        _NULL = Branch()

        if not self.deleted_branches:
            return _NULL
        if not self.create(curr_branch):
            return curr_branch

        if curr_branch == self.recombined_branch:
            if t >= self.start_time:
                return _NULL
            else:
                return self.source_branch
        elif curr_branch != self.merging_branch:
            return self.target_branch
        else:
            if t > self.deleted_node.time:
                return self._search_lower_node(self.deleted_node)
            else:
                return self._search_upper_node(self.deleted_node)

    # ------------------------------------------------------------------
    # Mutation remove/add helpers (called by ARG.remove / ARG.add)
    # ------------------------------------------------------------------

    def remove(
        self,
        prev_removed_branch: "Branch",
        next_removed_branch: "Branch",
        prev_split_branch: "Branch",
        next_split_branch: "Branch",
        cut_node: Optional["Node"] = None,
    ) -> None:
        """Update this recombination when the surrounding topology is pruned.

        Mirrors Recombination::remove (both overloads).
        """
        from .branch import Branch

        _NULL = Branch()

        if not self.deleted_branches and not self.inserted_branches:
            return

        if cut_node is not None:
            # Overload with cut_node
            if prev_removed_branch == next_removed_branch and \
               prev_split_branch == next_split_branch:
                return

            if prev_removed_branch == _NULL:
                self._break_front(next_removed_branch, next_split_branch, cut_node)
                return
            elif next_removed_branch == _NULL:
                self._break_end(prev_removed_branch, prev_split_branch, cut_node)
                return

            self._add_deleted(prev_split_branch)
            self._add_deleted(next_removed_branch)
            self._add_deleted(Branch(next_split_branch.lower_node, next_removed_branch.upper_node))
            self._add_deleted(Branch(next_removed_branch.upper_node, next_split_branch.upper_node))
            self._add_inserted(next_split_branch)
            self._add_inserted(prev_removed_branch)
            self._add_inserted(Branch(prev_split_branch.lower_node, prev_removed_branch.upper_node))
            self._add_inserted(Branch(prev_removed_branch.upper_node, prev_split_branch.upper_node))
            self._add_deleted(Branch(prev_removed_branch.lower_node, cut_node))
            self._add_inserted(Branch(next_removed_branch.lower_node, cut_node))
            self._simplify_branches()

            # Fix source_branch if it was destroyed
            self._fix_source_after_remove(prev_split_branch, prev_removed_branch)
            self._find_nodes()
            self._find_target_branch()
            if self.deleted_branches:
                self._find_recomb_info()
        else:
            # Overload without cut_node
            self._add_deleted(prev_split_branch)
            self._add_deleted(next_removed_branch)
            self._add_deleted(Branch(next_split_branch.lower_node, next_removed_branch.upper_node))
            self._add_deleted(Branch(next_removed_branch.upper_node, next_split_branch.upper_node))
            self._add_inserted(next_split_branch)
            self._add_inserted(prev_removed_branch)
            self._add_inserted(Branch(prev_split_branch.lower_node, prev_removed_branch.upper_node))
            self._add_inserted(Branch(prev_removed_branch.upper_node, prev_split_branch.upper_node))
            self._simplify_branches()
            if not self.deleted_branches and not self.inserted_branches:
                return
            self._fix_source_after_remove(prev_split_branch, prev_removed_branch)
            self._find_nodes()
            self._find_target_branch()
            self._find_recomb_info()

    def add(
        self,
        prev_added_branch: "Branch",
        next_added_branch: "Branch",
        prev_joining_branch: "Branch",
        next_joining_branch: "Branch",
        cut_node: Optional["Node"] = None,
    ) -> None:
        """Update this recombination when a new lineage is threaded in.

        Mirrors Recombination::add.
        """
        from .branch import Branch

        _NULL = Branch()

        if prev_added_branch == next_added_branch and \
           prev_joining_branch == next_joining_branch:
            return

        if next_added_branch != _NULL:
            self._add_inserted(next_added_branch)
            self._add_inserted(Branch(next_joining_branch.lower_node, next_added_branch.upper_node))
            self._add_inserted(Branch(next_added_branch.upper_node, next_joining_branch.upper_node))
            self._add_deleted(next_joining_branch)
            if cut_node is not None:
                self._add_deleted(Branch(next_added_branch.lower_node, cut_node))

        if prev_added_branch != _NULL:
            self._add_deleted(prev_added_branch)
            self._add_deleted(Branch(prev_joining_branch.lower_node, prev_added_branch.upper_node))
            self._add_deleted(Branch(prev_added_branch.upper_node, prev_joining_branch.upper_node))
            self._add_inserted(prev_joining_branch)
            if cut_node is not None:
                self._add_inserted(Branch(prev_added_branch.lower_node, cut_node))

        self._simplify_branches()

        if self.pos == 0:
            return
        if not self.deleted_branches:
            return

        self._find_nodes()

        # Update source_branch — mirrors C++ Recombination::add (lines 200-208):
        #   if (prev_joining == source_branch) {
        #       if (prev_added.upper == next_added.upper)   // pointer equality
        #           source_branch = Branch(prev_added.upper, source_branch.upper);
        #       else
        #           source_branch = Branch(source_branch.lower, prev_added.upper);
        #   } else { source_branch = search_lower_node(source_branch.lower); }
        if prev_joining_branch == self.source_branch:
            if prev_added_branch.upper_node is next_added_branch.upper_node:
                self.source_branch = Branch(prev_added_branch.upper_node, self.source_branch.upper_node)
            else:
                self.source_branch = Branch(self.source_branch.lower_node, prev_added_branch.upper_node)
        else:
            self.source_branch = self._search_lower_node(self.source_branch.lower_node)

        self._find_target_branch()
        self._find_recomb_info()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fix_source_after_remove(
        self,
        prev_split_branch: "Branch",
        prev_removed_branch: "Branch",
    ) -> None:
        from .branch import Branch

        destroyed1 = Branch(prev_split_branch.lower_node, prev_removed_branch.upper_node)
        destroyed2 = Branch(prev_removed_branch.upper_node, prev_split_branch.upper_node)
        if self.source_branch == destroyed1 or self.source_branch == destroyed2:
            self.source_branch = prev_split_branch

    def _break_front(
        self,
        next_removed: "Branch",
        next_split: "Branch",
        cut_node: "Node",
    ) -> None:
        from .branch import Branch

        self._add_deleted(next_removed)
        self._add_deleted(Branch(next_split.lower_node, next_removed.upper_node))
        self._add_deleted(Branch(next_removed.upper_node, next_split.upper_node))
        self._add_inserted(next_split)
        self._add_inserted(Branch(next_removed.lower_node, cut_node))
        self._simplify_branches()

    def _break_end(
        self,
        prev_removed: "Branch",
        prev_split: "Branch",
        cut_node: "Node",
    ) -> None:
        from .branch import Branch

        self._add_inserted(prev_removed)
        self._add_inserted(Branch(prev_split.lower_node, prev_removed.upper_node))
        self._add_inserted(Branch(prev_removed.upper_node, prev_split.upper_node))
        self._add_deleted(prev_split)
        self._add_deleted(Branch(prev_removed.lower_node, cut_node))
        self._simplify_branches()

    def _simplify_branches(self) -> None:
        """Remove branches that appear in both deleted and inserted sets."""
        common = self.deleted_branches & self.inserted_branches
        self.deleted_branches -= common
        self.inserted_branches -= common

    def _add_deleted(self, b: "Branch") -> None:
        if b.upper_node is not None and b.lower_node is not None:
            self.deleted_branches.add(b)

    def _add_inserted(self, b: "Branch") -> None:
        if b.upper_node is not None and b.lower_node is not None:
            self.inserted_branches.add(b)

    def _find_nodes(self) -> None:
        """Identify inserted_node and deleted_node from branch sets."""
        prev_upper = {b.upper_node for b in self.deleted_branches}
        next_upper = {b.upper_node for b in self.inserted_branches}
        for n in prev_upper:
            if n not in next_upper:
                self.deleted_node = n
        for n in next_upper:
            if n not in prev_upper:
                self.inserted_node = n

    def _find_target_branch(self) -> None:
        """Find the branch that the recombined lineage joins."""
        from .branch import Branch

        if self.inserted_node is None:
            self.target_branch = Branch()
            return

        t = self.inserted_node.time
        for b in self.deleted_branches:
            if b == self.source_branch:
                continue
            if b.lower_node.time > t or b.upper_node.time < t:
                continue
            lower = Branch(b.lower_node, self.inserted_node)
            upper = Branch(self.inserted_node, b.upper_node)
            if lower in self.inserted_branches and upper in self.inserted_branches:
                self.target_branch = b
                return

        # Fallback: accept partial match
        for b in self.deleted_branches:
            if b == self.source_branch:
                continue
            if b.lower_node.time > t or b.upper_node.time < t:
                continue
            lower = Branch(b.lower_node, self.inserted_node)
            upper = Branch(self.inserted_node, b.upper_node)
            if lower in self.inserted_branches or upper in self.inserted_branches:
                self.target_branch = b
                return

        self.target_branch = Branch()

    def _find_recomb_info(self) -> None:
        """Compute start_time, merging_branch, transfer branches, etc."""
        from .branch import Branch

        _INT_MAX = sys.maxsize
        if self.pos == 0 or self.pos == _INT_MAX:
            return

        dn = self.deleted_node
        l = None
        u = None

        for b in self.deleted_branches:
            if b == self.source_branch:
                continue
            if b.upper_node is dn:
                l = self.inserted_node if b == self.target_branch else b.lower_node
            elif b.lower_node is dn:
                u = self.inserted_node if b == self.target_branch else b.upper_node

        self.merging_branch = Branch(l, u)
        self.recombined_branch = Branch(
            self.source_branch.lower_node, self.inserted_node
        )
        self.source_sister_branch = self._search_upper_node(dn)
        self.source_parent_branch = self._search_lower_node(dn)

        # Transfer branches
        candidate_lower = Branch(self.target_branch.lower_node, self.inserted_node)
        if candidate_lower in self.inserted_branches:
            self.lower_transfer_branch = candidate_lower
        else:
            self.lower_transfer_branch = self.merging_branch

        candidate_upper = Branch(self.inserted_node, self.target_branch.upper_node)
        if candidate_upper in self.inserted_branches:
            self.upper_transfer_branch = candidate_upper
        else:
            self.upper_transfer_branch = self.merging_branch

        # start_time is set by approx_sample_recombinations / sample_recombination,
        # NOT here — mirroring C++ find_recomb_info which does not touch start_time.

    def _search_upper_node(self, n: "Node") -> "Branch":
        """Find the deleted branch (not source) with upper_node == n."""
        from .branch import Branch

        for b in self.deleted_branches:
            if b != self.source_branch and b.upper_node is n:
                return b
        return Branch()

    def _search_lower_node(self, n: "Node") -> "Branch":
        """Find the deleted branch with lower_node == n."""
        from .branch import Branch

        for b in self.deleted_branches:
            if b.lower_node is n:
                return b
        return Branch()

    def __repr__(self) -> str:
        return (
            f"Recombination(pos={self.pos}, "
            f"|del|={len(self.deleted_branches)}, "
            f"|ins|={len(self.inserted_branches)})"
        )
