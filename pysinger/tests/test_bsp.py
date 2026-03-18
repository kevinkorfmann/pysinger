"""Tests for BSP forward HMM."""
import math
import pytest

from pysinger.data.node import Node
from pysinger.data.branch import Branch
from pysinger.hmm.bsp import BSP
from pysinger.hmm.emission import BinaryEmission


def make_two_branch_set():
    """Two branches spanning [0,1] and [0,3]."""
    n0 = Node(time=0.0, index=0)
    n1 = Node(time=1.0, index=1)
    n2 = Node(time=3.0, index=2)
    root = Node(time=math.inf, index=-1)
    b1 = Branch(n0, n1)
    b2 = Branch(n1, n2)
    b3 = Branch(n2, root)
    return {b1, b2, b3}, n0, n1, n2, root


class TestBSP:
    def setup_method(self):
        self.bsp = BSP()
        self.bsp.set_emission(BinaryEmission())

    def test_start_initialises_intervals(self):
        branches, n0, n1, n2, root = make_two_branch_set()
        self.bsp.start(branches, t=0.0)
        assert len(self.bsp.curr_intervals) > 0

    def test_forward_prob_nonneg(self):
        branches, n0, n1, n2, root = make_two_branch_set()
        self.bsp.start(branches, t=0.0)
        self.bsp.null_emit(theta=0.1, query_node=n0)
        self.bsp.forward(rho=0.01)
        fp = self.bsp.forward_probs[self.bsp.curr_index]
        assert all(p >= 0 for p in fp)

    def test_null_emit_normalises(self):
        branches, n0, n1, n2, root = make_two_branch_set()
        self.bsp.start(branches, t=0.0)
        self.bsp.null_emit(theta=0.1, query_node=n0)
        fp = self.bsp.forward_probs[self.bsp.curr_index]
        assert abs(sum(fp) - 1.0) < 1e-9

    def test_forward_step_increases_index(self):
        branches, n0, n1, n2, root = make_two_branch_set()
        self.bsp.start(branches, t=0.0)
        before = self.bsp.curr_index
        self.bsp.null_emit(theta=0.1, query_node=n0)
        self.bsp.forward(rho=0.01)
        assert self.bsp.curr_index == before + 1

    def test_sample_joining_branches_returns_dict(self):
        import numpy as np
        from sortedcontainers import SortedDict
        branches, n0, n1, n2, root = make_two_branch_set()
        self.bsp.set_rng(np.random.default_rng(42))
        self.bsp.start(branches, t=0.0)
        # Run 3 steps
        coords = [0.0, 100.0, 200.0, 300.0, 400.0]
        for i in range(3):
            self.bsp.null_emit(theta=0.1, query_node=n0)
            self.bsp.forward(rho=0.01)
        self.bsp.null_emit(theta=0.1, query_node=n0)
        jb = self.bsp.sample_joining_branches(start_index=0, coordinates=coords)
        assert isinstance(jb, SortedDict)
        assert len(jb) >= 1
        # All values should be Branch objects
        for v in jb.values():
            assert isinstance(v, Branch)
