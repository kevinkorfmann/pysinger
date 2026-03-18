"""
FitchReconstruction — Fitch parsimony ancestral state reconstruction.

Mirrors Fitch_reconstruction.cpp / Fitch_reconstruction.hpp.

Given a Tree, performs a two-pass Fitch parsimony reconstruction at each
genomic position to assign ancestral states to internal nodes.

Pass 1 (pruning / up-pass): bottom-up, assigns a state to each node based
on the intersection (or union when intersection is empty) of its children.
We encode ambiguity as 0.5.

Pass 2 (peeling / down-pass): top-down, resolves ambiguous states by
propagating the parent's definite state downward.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.node import Node
    from ..data.tree import Tree
    from ..data.recombination import Recombination


class FitchReconstruction:
    """Two-pass Fitch parsimony reconstruction on a Tree.

    Usage::

        fr = FitchReconstruction(tree)
        fr.reconstruct(pos)      # assigns states for position *pos*
        fr.update(recombination) # advance tree topology
    """

    def __init__(self, tree: "Tree") -> None:
        self.base_tree = tree
        # Map node → (child1, child2)
        self.children_nodes: Dict["Node", Tuple[Optional["Node"], Optional["Node"]]] = {}
        # Map node → parent
        self.parent_node: Dict["Node", "Node"] = {}
        # All non-root nodes
        self.node_set: Set["Node"] = set()

        self.pruning_node_states: Dict["Node", float] = {}
        self.peeling_node_states: Dict["Node", float] = {}
        self.recon_pos: float = 0.0

        self._fill_tree_info(tree)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(self, pos: float) -> None:
        """Perform Fitch reconstruction at genomic position *pos*.

        Writes the inferred ancestral states back into each node via
        ``node.write_state(pos, state)``.

        Mirrors Fitch_reconstruction::reconstruct.
        """
        self.recon_pos = pos
        self.pruning_node_states = {}
        self.peeling_node_states = {}

        for n in self.node_set:
            self._pruning_pass(n)
        for n in self.node_set:
            self._peeling_pass(n)
        for n, s in self.peeling_node_states.items():
            n.write_state(pos, s)

    def update(self, r: "Recombination") -> None:
        """Advance the topology by applying recombination *r*.

        Mirrors Fitch_reconstruction::update.
        """
        self.base_tree.forward_update(r)
        self._fill_tree_info(self.base_tree)

    # ------------------------------------------------------------------
    # Private: tree info construction
    # ------------------------------------------------------------------

    def _fill_tree_info(self, tree: "Tree") -> None:
        """Build children_nodes, parent_node, node_set from *tree*.

        Mirrors Fitch_reconstruction::fill_tree_info.
        """
        self.node_set = set()
        self.children_nodes = {}
        self.parent_node = {}

        for child, parent in tree.parents.items():
            self.node_set.add(child)
            self.parent_node[child] = parent
            if parent in self.children_nodes:
                # Second child
                c1, _ = self.children_nodes[parent]
                self.children_nodes[parent] = (c1, child)
            else:
                self.children_nodes[parent] = (child, None)

    # ------------------------------------------------------------------
    # Private: Fitch up/down
    # ------------------------------------------------------------------

    def _fitch_up(
        self,
        c1: Optional["Node"],
        c2: Optional["Node"],
        p: "Node",
    ) -> None:
        """Merge children states upward into parent.

        Mirrors Fitch_reconstruction::Fitch_up.
        """
        if c1 is None or c2 is None:
            return
        s1 = self.pruning_node_states[c1]
        s2 = self.pruning_node_states[c2]
        if s1 == 0.5:
            s = s2
        elif s2 == 0.5:
            s = s1
        elif s1 != s2:
            s = 0.5  # ambiguous (union)
        else:
            s = s1
        self.pruning_node_states[p] = s

    def _fitch_down(self, parent: "Node", child: "Node") -> None:
        """Propagate parent state downward to child.

        Mirrors Fitch_reconstruction::Fitch_down.
        """
        if parent.index == -1:
            # Root sentinel: resolve ambiguity to 0
            top_state = self.pruning_node_states[child]
            s = 0.0 if top_state == 0.5 else top_state
            self.peeling_node_states[child] = s
            return
        sp = self.peeling_node_states[parent]
        sc = self.pruning_node_states[child]
        if sp in (0.0, 1.0):
            # Parent is definite: resolve child ambiguity with parent state
            s = sp if sc == 0.5 else sc
        else:
            s = sc
        self.peeling_node_states[child] = s

    # ------------------------------------------------------------------
    # Private: recursive passes
    # ------------------------------------------------------------------

    def _pruning_pass(self, n: "Node") -> None:
        """Bottom-up pass: compute pruning state for *n*.

        Mirrors Fitch_reconstruction::pruning_pass.
        """
        if n in self.pruning_node_states:
            return
        if n not in self.children_nodes:
            # Leaf: read current state
            self.pruning_node_states[n] = n.get_state(self.recon_pos)
            return
        c1, c2 = self.children_nodes[n]
        if c1 is not None:
            self._pruning_pass(c1)
        if c2 is not None:
            self._pruning_pass(c2)
        self._fitch_up(c1, c2, n)

    def _peeling_pass(self, n: "Node") -> None:
        """Top-down pass: compute peeling state for *n*.

        Mirrors Fitch_reconstruction::peeling_pass.
        """
        if n in self.peeling_node_states:
            return
        if n not in self.parent_node:
            # Root or disconnected: take pruning state
            self.peeling_node_states[n] = self.pruning_node_states.get(n, 0.0)
            return
        p = self.parent_node[n]
        if p.index == -1:
            # Parent is root sentinel
            top_state = self.pruning_node_states.get(n, 0.5)
            self.peeling_node_states[n] = 0.0 if top_state == 0.5 else top_state
        elif p.index == -2:
            # Special sentinel index -2: pass through pruning state
            self.peeling_node_states[n] = self.pruning_node_states.get(n, 0.0)
        else:
            self._peeling_pass(p)
            self._fitch_down(p, n)
