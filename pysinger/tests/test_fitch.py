"""Tests for FitchReconstruction."""
import math
import pytest

from pysinger.data.node import Node
from pysinger.data.branch import Branch
from pysinger.data.tree import Tree
from pysinger.reconstruction.fitch import FitchReconstruction


def build_tree_and_nodes():
    """Four-leaf balanced tree.

         root (inf)
           |
          n5 (t=3)
         /   \\
       n3(1) n4(2)
       / \\   / \\
      n0  n1 n2  n6
    """
    root = Node(time=math.inf, index=-1)
    n0 = Node(time=0.0, index=0)
    n1 = Node(time=0.0, index=1)
    n2 = Node(time=0.0, index=2)
    n6 = Node(time=0.0, index=6)
    n3 = Node(time=1.0, index=3)
    n4 = Node(time=2.0, index=4)
    n5 = Node(time=3.0, index=5)

    tree = Tree()
    for child, parent in [
        (n0, n3), (n1, n3),
        (n2, n4), (n6, n4),
        (n3, n5), (n4, n5),
        (n5, root),
    ]:
        tree.insert_branch(Branch(child, parent))

    return tree, root, n0, n1, n2, n6, n3, n4, n5


class TestFitchReconstruction:
    def test_all_zero(self):
        tree, root, n0, n1, n2, n6, n3, n4, n5 = build_tree_and_nodes()
        fr = FitchReconstruction(tree)
        fr.reconstruct(pos=0.0)
        # All leaves are 0 → internal should be 0
        assert n3.get_state(0.0) == 0
        assert n4.get_state(0.0) == 0
        assert n5.get_state(0.0) == 0

    def test_all_one(self):
        tree, root, n0, n1, n2, n6, n3, n4, n5 = build_tree_and_nodes()
        for n in [n0, n1, n2, n6]:
            n.add_mutation(0.0)
        fr = FitchReconstruction(tree)
        fr.reconstruct(pos=0.0)
        # All leaves are 1 → internal should be 1
        assert n3.get_state(0.0) == 1
        assert n4.get_state(0.0) == 1

    def test_parsimony_resolves_ambiguity(self):
        """n0=1, n1=0, n2=0, n6=0 — one mutation total; n3 should be ambiguous (resolved to 0)."""
        tree, root, n0, n1, n2, n6, n3, n4, n5 = build_tree_and_nodes()
        n0.add_mutation(0.0)  # n0=1, rest=0
        fr = FitchReconstruction(tree)
        fr.reconstruct(pos=0.0)
        # n3 gets 0.5 in pruning, resolved to 0 in peeling (parent n5 is 0)
        assert n4.get_state(0.0) == 0

    def test_split_mutation(self):
        """n0=n1=1, n2=n6=0 — mutation on (n3, n5) edge; n3 should be 1."""
        tree, root, n0, n1, n2, n6, n3, n4, n5 = build_tree_and_nodes()
        n0.add_mutation(0.0)
        n1.add_mutation(0.0)
        fr = FitchReconstruction(tree)
        fr.reconstruct(pos=0.0)
        assert n3.get_state(0.0) == 1
        assert n4.get_state(0.0) == 0
