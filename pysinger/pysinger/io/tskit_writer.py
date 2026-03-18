"""
tskit writer — convert a pysinger ARG to a tskit TreeSequence.

Mirrors the conceptual output of SINGER's write() method, but produces
a tskit.TreeSequence that can be analysed with the tskit/msprime ecosystem.

The conversion:
  1. Add all non-root nodes (time = node.time * 2*Ne) as tskit individuals.
  2. Walk the ARG recombinations to build a list of (left, right, parent, child)
     edges spanning each genomic interval.
  3. Sort edges (required by tskit) and call tables.sort() + tables.tree_sequence().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.arg import ARG

try:
    import tskit
    _HAS_TSKIT = True
except ImportError:
    _HAS_TSKIT = False


def arg_to_tskit(arg: "ARG", Ne: float = 1.0):  # -> tskit.TreeSequence
    """Convert *arg* to a :class:`tskit.TreeSequence`.

    Parameters
    ----------
    arg : ARG
        The pysinger ARG object.
    Ne : float
        Effective population size used to convert coalescent time units
        to generations (t_generations = t_coalescent * 2 * Ne).

    Returns
    -------
    ts : tskit.TreeSequence
    """
    if not _HAS_TSKIT:
        raise ImportError("tskit is required for arg_to_tskit(). Install it with: pip install tskit")

    tables = tskit.TableCollection(sequence_length=arg.sequence_length)

    # ------------------------------------------------------------------
    # 1. Discover all nodes by walking the full tree sequence
    # ------------------------------------------------------------------
    # arg.node_set may be empty (add_node() is not always called), so we
    # discover nodes by replaying all recombinations and collecting every
    # node that appears in a parent/child relationship.
    node_map = {}
    seen_nodes = set()  # use id() to avoid hashing issues
    seen_by_id = {}     # id(node) → node

    def _collect(tree_obj):
        for child, parent in tree_obj.parents.items():
            for n in (child, parent):
                if n is not None and n.index != -1 and id(n) not in seen_nodes:
                    seen_nodes.add(id(n))
                    seen_by_id[id(n)] = n

    # Walk the whole genome
    tree = arg.get_tree_at(0.0)
    _collect(tree)
    for pos in arg.recombinations.keys():
        if 0 < pos < arg.sequence_length:
            r = arg.recombinations[pos]
            tree.forward_update(r)
            _collect(tree)

    all_nodes = sorted(seen_by_id.values(), key=lambda n: (n.time, n.index))

    for n in all_nodes:
        if n.index == -1:
            continue  # skip root sentinel
        is_sample = n in arg.sample_nodes
        tskit_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE if is_sample else 0,
            time=n.time * Ne,  # 1 SINGER unit = Ne generations (haploid coalescent)
        )
        node_map[id(n)] = tskit_id

    # ------------------------------------------------------------------
    # 2. Walk trees and emit edges
    # ------------------------------------------------------------------
    # Replay recombinations left to right over the full sequence; at each
    # topology change emit edges for the current tree interval.
    seq_len = arg.sequence_length
    tree = arg.get_tree_at(0.0)
    recomb_positions = sorted(
        pos for pos in arg.recombinations.keys()
        if 0 < pos < seq_len
    )

    prev_pos = 0.0

    def _emit_edges(tree_obj, left: float, right: float) -> None:
        if left >= right:
            return
        for child, parent in tree_obj.parents.items():
            if parent.index == -1:
                continue  # skip edges to root sentinel
            if parent.time <= child.time:
                continue  # skip time-ordering violations (malformed ARG state)
            c_id = node_map.get(id(child))
            p_id = node_map.get(id(parent))
            if c_id is None or p_id is None:
                continue
            tables.edges.add_row(left=left, right=right, parent=p_id, child=c_id)

    for rpos in recomb_positions:
        _emit_edges(tree, prev_pos, rpos)
        r = arg.recombinations[rpos]
        tree.forward_update(r)
        prev_pos = rpos

    _emit_edges(tree, prev_pos, seq_len)

    # ------------------------------------------------------------------
    # 3. Sort and build tree sequence
    # ------------------------------------------------------------------
    tables.sort()
    ts = tables.tree_sequence()
    return ts
