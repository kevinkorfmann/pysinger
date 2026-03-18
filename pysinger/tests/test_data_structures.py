"""Tests for data layer: Node, Branch, Tree, Recombination, Interval."""
import math
import pytest

from pysinger.data.node import Node
from pysinger.data.branch import Branch
from pysinger.data.tree import Tree
from pysinger.data.interval import Interval, IntervalInfo


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNode:
    def test_default_state(self):
        n = Node(time=0.0)
        assert n.get_state(500.0) == 0

    def test_add_and_get_mutation(self):
        n = Node(time=0.0)
        n.add_mutation(100.0)
        assert n.get_state(100.0) == 1
        assert n.get_state(99.9) == 0
        assert n.get_state(200.0) == 0   # exact match only; 200 != 100

    def test_write_state(self):
        n = Node(time=0.0)
        n.write_state(200.0, 1.0)
        assert n.get_state(200.0) == 1.0
        n.write_state(200.0, 0.0)
        assert n.get_state(200.0) == 0

    def test_identity_hash(self):
        n1 = Node(time=0.0, index=1)
        n2 = Node(time=0.0, index=1)
        assert n1 != n2   # different objects
        assert hash(n1) != hash(n2)
        s = {n1, n2}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

class TestBranch:
    def test_null_branch(self):
        b = Branch()
        assert b.is_null()

    def test_non_null_branch(self):
        n0 = Node(time=0.0)
        n1 = Node(time=1.0)
        b = Branch(n0, n1)
        assert not b.is_null()
        assert b.length == 1.0

    def test_identity_equality(self):
        n0 = Node(time=0.0)
        n1 = Node(time=1.0)
        n2 = Node(time=1.0)
        b1 = Branch(n0, n1)
        b2 = Branch(n0, n1)  # same node objects → equal (mirrors C++ pointer equality)
        b3 = Branch(n0, n2)  # different upper node → not equal
        assert b1 == b2
        assert b1 != b3
        assert b1 == b1

    def test_hashable_in_set(self):
        n0 = Node(time=0.0)
        n1 = Node(time=1.0)
        b = Branch(n0, n1)
        s = {b}
        assert b in s


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class TestTree:
    def test_insert_delete(self, simple_nodes, simple_tree):
        root, n0, n1, n2, n3, n4, n5 = simple_nodes
        tree = simple_tree
        assert n0 in tree.parents
        assert n3 in tree.children
        # delete n0 → n3 edge
        b = Branch(n0, n3)
        tree.delete_branch(b)
        assert n0 not in tree.parents

    def test_find_sibling(self, simple_nodes, simple_tree):
        root, n0, n1, n2, n3, n4, n5 = simple_nodes
        sib = simple_tree.find_sibling(n0)
        assert sib is n1


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

class TestInterval:
    def test_fill_time_midpoint(self):
        n0 = Node(time=0.0)
        n1 = Node(time=2.0)
        b = Branch(n0, n1)
        iv = Interval(b, 0.5, 1.5, 0)
        iv.fill_time()
        assert 0.5 <= iv.time <= 1.5

    def test_full(self):
        n0 = Node(time=0.0)
        n1 = Node(time=2.0)
        b = Branch(n0, n1)
        iv = Interval(b, 0.0, 2.0, 0)
        assert iv.full(0.0)
        iv2 = Interval(b, 0.5, 1.5, 0)
        assert not iv2.full(0.0)

    def test_interval_info_hashable(self):
        n0 = Node(time=0.0)
        n1 = Node(time=2.0)
        b = Branch(n0, n1)
        info = IntervalInfo(b, 0.5, 1.5)
        d = {info: 1.0}
        assert d[info] == 1.0
