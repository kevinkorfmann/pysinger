"""
Tree — a marginal coalescent tree at a single genomic position.

Mirrors Tree.cpp / Tree.hpp.

The Tree is represented by two dicts:
  parents:  child_node  -> parent_node
  children: parent_node -> set of child_nodes

The topology is updated forward/backward along the genome by applying
Recombination records.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .branch import Branch
    from .node import Node
    from .recombination import Recombination


class Tree:
    """Marginal coalescent tree at one genomic position.

    Attributes
    ----------
    parents:  child  → parent mapping (all non-root nodes present as keys).
    children: parent → set of children (only internal nodes as keys).
    """

    def __init__(self) -> None:
        self.parents: Dict["Node", "Node"] = {}
        self.children: Dict["Node", Set["Node"]] = {}

    # ------------------------------------------------------------------
    # Basic branch operations
    # ------------------------------------------------------------------

    def insert_branch(self, branch: "Branch") -> None:
        """Add *branch* to the tree."""
        ln, un = branch.lower_node, branch.upper_node
        assert ln is not None and un is not None
        self.parents[ln] = un
        if un not in self.children:
            self.children[un] = set()
        self.children[un].add(ln)

    def delete_branch(self, branch: "Branch") -> None:
        """Remove *branch* from the tree."""
        ln, un = branch.lower_node, branch.upper_node
        assert ln is not None and un is not None
        self.parents.pop(ln, None)
        if un in self.children:
            ch = self.children[un]
            ch.discard(ln)
            if not ch:
                del self.children[un]

    # ------------------------------------------------------------------
    # Topology updates
    # ------------------------------------------------------------------

    def forward_update(self, r: "Recombination") -> None:
        """Apply recombination *r* moving right along the genome."""
        for b in r.deleted_branches:
            self.delete_branch(b)
        for b in r.inserted_branches:
            self.insert_branch(b)

    def backward_update(self, r: "Recombination") -> None:
        """Undo recombination *r* moving left along the genome."""
        for b in r.inserted_branches:
            self.delete_branch(b)
        for b in r.deleted_branches:
            self.insert_branch(b)

    def internal_forward_update(self, r: "Recombination", cut_time: float) -> None:
        """forward_update restricted to branches with upper_node.time > cut_time."""
        for b in r.deleted_branches:
            if b.upper_node.time > cut_time:
                self.delete_branch(b)
        for b in r.inserted_branches:
            if b.upper_node.time > cut_time:
                self.insert_branch(b)

    def internal_backward_update(self, r: "Recombination", cut_time: float) -> None:
        """backward_update restricted to branches with upper_node.time > cut_time."""
        for b in r.inserted_branches:
            if b.upper_node.time > cut_time:
                self.delete_branch(b)
        for b in r.deleted_branches:
            if b.upper_node.time > cut_time:
                self.insert_branch(b)

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------

    def find_sibling(self, n: "Node") -> Optional["Node"]:
        """Return the sibling of *n* (the other child of n's parent)."""
        p = self.parents[n]
        ch = self.children[p]
        for c in ch:
            if c is not n:
                return c
        return None  # shouldn't happen in a binary tree

    def find_joining_branch(self, removed_branch: "Branch") -> "Branch":
        """Return Branch(sibling, grandparent) for the removed branch.

        Mirrors Tree::find_joining_branch.
        """
        from .branch import Branch

        if removed_branch.is_null():
            return Branch()
        if removed_branch.upper_node not in self.parents:
            if removed_branch.lower_node not in self.parents:
                return Branch()
            # upper_node is the root of the current tree (has no parent)
            c = self.find_sibling(removed_branch.lower_node)
            if c is None:
                return Branch()
            return Branch(c, removed_branch.upper_node)
        p = self.parents[removed_branch.upper_node]
        if removed_branch.lower_node not in self.parents:
            return Branch()
        c = self.find_sibling(removed_branch.lower_node)
        if c is None:
            return Branch()
        return Branch(c, p)

    # ------------------------------------------------------------------
    # MCMC remove / add (used to update cut_tree in ARG)
    # ------------------------------------------------------------------

    def remove(self, branch: "Branch", cut_node: "Node") -> None:
        """Remove *branch* from the tree and replace with cut lineage.

        Mirrors Tree::remove.  After this call the tree has `cut_node` as
        a leaf connected to branch.lower_node, and the sibling is directly
        connected to branch.upper_node's parent.
        """
        from .branch import Branch

        assert branch.upper_node.index >= 0, "Cannot remove branch to root sentinel"
        joining_branch = self.find_joining_branch(branch)
        sibling = self.find_sibling(branch.lower_node)
        parent = self.parents[branch.upper_node]
        sibling_branch = Branch(sibling, branch.upper_node)
        parent_branch = Branch(branch.upper_node, parent)
        cut_branch = Branch(branch.lower_node, cut_node)

        self.delete_branch(branch)
        self.delete_branch(sibling_branch)
        self.delete_branch(parent_branch)
        self.insert_branch(joining_branch)
        self.insert_branch(cut_branch)

    def add(
        self,
        added_branch: "Branch",
        joining_branch: "Branch",
        cut_node: Optional["Node"],
    ) -> None:
        """Insert *added_branch* joining at *joining_branch*.

        Mirrors Tree::add.
        """
        from .branch import Branch

        lower_branch = Branch(joining_branch.lower_node, added_branch.upper_node)
        upper_branch = Branch(added_branch.upper_node, joining_branch.upper_node)
        self.delete_branch(joining_branch)
        self.insert_branch(lower_branch)
        self.insert_branch(upper_branch)
        self.insert_branch(added_branch)
        if cut_node is not None:
            cut_branch = Branch(added_branch.lower_node, cut_node)
            self.delete_branch(cut_branch)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def length(self) -> float:
        """Total branch length (excluding root sentinel branches)."""
        total = 0.0
        for child, parent in self.parents.items():
            if parent.index != -1:
                total += parent.time - child.time
        return total

    def copy(self) -> "Tree":
        """Shallow-copy the topology dicts (nodes are shared)."""
        t = Tree()
        t.parents = dict(self.parents)
        t.children = {p: set(ch) for p, ch in self.children.items()}
        return t

    def get_branches(self) -> Set["Branch"]:
        """Return all branches in the tree as a set of Branch objects."""
        from .branch import Branch
        return {Branch(child, parent) for child, parent in self.parents.items()}

    def __iter__(self):
        """Allow passing a Tree directly where a set of branches is expected."""
        from .branch import Branch
        for child, parent in self.parents.items():
            yield Branch(child, parent)

    def __repr__(self) -> str:
        return f"Tree(n_branches={len(self.parents)})"
