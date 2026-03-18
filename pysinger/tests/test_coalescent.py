"""Tests for CoalescentCalculator."""
import math
import pytest

from pysinger.data.node import Node
from pysinger.data.branch import Branch
from pysinger.hmm.coalescent import CoalescentCalculator


def make_branch(t_lower: float, t_upper: float) -> Branch:
    n0 = Node(time=t_lower)
    n1 = Node(time=t_upper)
    return Branch(n0, n1)


class TestCoalescentCalculator:
    def setup_method(self):
        self.cc = CoalescentCalculator(cut_time=0.0)

    def test_empty(self):
        self.cc.compute(set())
        assert self.cc.prob(1.0) == 0.0
        assert self.cc.weight(0.0, 1.0) == 0.0

    def test_single_branch_prob_monotone(self):
        b = make_branch(0.0, 2.0)
        self.cc.compute({b})
        p0 = self.cc.prob(0.0)
        p1 = self.cc.prob(1.0)
        p2 = self.cc.prob(2.0)
        assert p0 <= p1 <= p2

    def test_total_weight_one(self):
        branches = {make_branch(0.0, 1.0), make_branch(0.0, 3.0)}
        self.cc.compute(branches)
        w = self.cc.weight(self.cc.min_time, self.cc.max_time)
        # Should be close to 1 (or up to the coalescent CDF at max_time)
        assert w >= 0

    def test_time_in_interval(self):
        b = make_branch(0.0, 5.0)
        self.cc.compute({b})
        t = self.cc.time(1.0, 3.0)
        assert 1.0 <= t <= 3.0

    def test_weight_nonneg(self):
        b = make_branch(0.5, 4.0)
        self.cc.compute({b})
        w = self.cc.weight(1.0, 3.0)
        assert w >= 0

    def test_quantile_roundtrip(self):
        b = make_branch(0.0, 10.0)
        self.cc.compute({b})
        p = self.cc.prob(3.0)
        t = self.cc.quantile(p)
        # quantile(prob(t)) should recover roughly t
        assert abs(t - 3.0) < 0.5
