"""
Shared fixtures for pysingerarg tests.
"""
import math
import pytest

from pysingerarg.data.node import Node
from pysingerarg.data.branch import Branch
from pysingerarg.data.tree import Tree
from pysingerarg.data.recombination import Recombination
from pysingerarg.data.arg import ARG


@pytest.fixture
def simple_nodes():
    """Four leaf nodes and two internal nodes forming a simple tree.

    Topology (time increases upward):
          root (inf)
           |
          n5 (t=3)
         /    \\
       n3(1)  n4(2)
       / \\    / \\
      n0  n1 n2  (leaf)
    (t=0)(t=0)(t=0)
    """
    root = Node(time=math.inf, index=-1)
    n0 = Node(time=0.0, index=0)
    n1 = Node(time=0.0, index=1)
    n2 = Node(time=0.0, index=2)
    n3 = Node(time=1.0, index=3)
    n4 = Node(time=2.0, index=4)
    n5 = Node(time=3.0, index=5)
    return root, n0, n1, n2, n3, n4, n5


@pytest.fixture
def simple_tree(simple_nodes):
    """A Tree with the topology from simple_nodes."""
    root, n0, n1, n2, n3, n4, n5 = simple_nodes
    tree = Tree()
    for child, parent in [(n0, n3), (n1, n3), (n2, n4), (n3, n5), (n4, n5), (n5, root)]:
        tree.insert_branch(Branch(child, parent))
    return tree


@pytest.fixture
def small_arg():
    """A minimal ARG with 2 sample nodes, 1 kb sequence."""
    n0 = Node(time=0.0, index=0)
    n1 = Node(time=0.0, index=1)
    n0.add_mutation(100.0)
    n1.add_mutation(400.0)
    arg = ARG(Ne=1.0, sequence_length=1000.0)
    arg.discretize(100.0)
    arg.build_singleton_arg(n0)
    arg.add_sample(n1)
    return arg, n0, n1


@pytest.fixture
def small_vcf_path(tmp_path):
    """Path to the small fixture VCF."""
    import os
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "small.vcf")
    return fixture
