"""
ARG — the Ancestral Recombination Graph, the central data structure.

Mirrors ARG.cpp / ARG.hpp.

An ARG is encoded as a sorted map of Recombination records keyed by
genomic position.  A marginal tree at position x is obtained by
replaying all records from position 0 up to (and including) x.

The two main MCMC operations are:
  remove(cut_point) — extract a single lineage from the ARG, leaving
                      the remaining haplotypes connected.
  add(joining, added) — thread the lineage back in at new positions.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sortedcontainers import SortedDict

from .branch import Branch
from .node import Node
from .recombination import Recombination
from .tree import Tree

_INT_MAX = sys.maxsize


def _choose_time(lb: float, ub: float) -> float:
    """Midpoint in log space; falls back to linear for small intervals.

    Mirrors RSP_smc::choose_time(double lb, double ub).
    """
    if ub - lb < 0.01:
        return 0.5 * (lb + ub)
    mt = math.log(0.5 * (math.exp(lb) + math.exp(ub)))
    return max(lb, min(ub, mt))


class ARG:
    """Ancestral Recombination Graph.

    Attributes
    ----------
    Ne:                   Effective population size (diploid).
    sequence_length:      Length of the genomic region (base pairs).
    root:                 Sentinel root node (time=inf, index=-1).
    sample_nodes:         Set of leaf (sample) nodes.
    node_set:             All non-root nodes.
    recombinations:       SortedDict[pos → Recombination].  Always has
                          sentinels at 0 and INT_MAX.
    mutation_sites:       Sorted set of positions carrying derived alleles.
    mutation_branches:    pos → set of Branches carrying the mutation.
    coordinates:          Grid of genomic positions (HMM bins).
    rhos:                 Scaled recombination rate per bin.
    thetas:               Scaled mutation rate per bin.

    MCMC working variables (reset by clear_remove_info):
    removed_branches:     pos → Branch that was removed.
    joining_branches:     pos → Branch that "jumped over" the cut.
    cut_tree:             Marginal tree at the cut position.
    cut_pos, cut_node:    Position and sentinel node for the cut.
    start, end:           Genomic extent of the current MCMC window.
    """

    def __init__(self, Ne: float = 1.0, sequence_length: float = 1.0) -> None:
        self.Ne = Ne
        self.sequence_length = sequence_length

        # Root sentinel: time=inf, index=-1
        self.root = Node(time=math.inf, index=-1)

        self.sample_nodes: Set[Node] = set()
        self.node_set: Set[Node] = set()

        # Recombination records — always have sentinels at 0 and INT_MAX
        self.recombinations: SortedDict = SortedDict()
        r0 = Recombination()
        r0.set_pos(0.0)
        self.recombinations[0.0] = r0
        r_end = Recombination()
        r_end.set_pos(float(_INT_MAX))
        self.recombinations[float(_INT_MAX)] = r_end

        # Mutation data
        self.mutation_sites: SortedDict = SortedDict()  # pos → True (sorted set)
        self.mutation_sites[float(_INT_MAX)] = True
        self.mutation_branches: Dict[float, Set[Branch]] = {
            float(_INT_MAX): set()
        }

        # HMM grid
        self.coordinates: List[float] = []
        self.rhos: List[float] = []
        self.thetas: List[float] = []
        self.bin_size: float = 1.0
        self.bin_num: int = 0

        # MCMC working state
        self.removed_branches: SortedDict = SortedDict()  # pos → Branch
        self.joining_branches: SortedDict = SortedDict()  # pos → Branch
        self.cut_tree: Tree = Tree()
        self.start_tree: Tree = Tree()
        self.end_tree: Tree = Tree()
        self.cut_pos: float = 0.0
        self.cut_time: float = 0.0
        self.cut_node: Optional[Node] = None
        self.start: float = 0.0
        self.end: float = sequence_length

        # RNG (set by Sampler)
        self.rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Random helper
    # ------------------------------------------------------------------

    def _random(self) -> float:
        if self.rng is not None:
            return float(self.rng.uniform())
        return float(np.random.uniform())

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def add_sample(self, n: Node) -> None:
        """Register sample node *n* and update mutation_sites."""
        self.sample_nodes.add(n)
        for pos in n.mutation_sites:
            if pos != -1:
                self.mutation_sites[pos] = True
        self.removed_branches = SortedDict()
        self.removed_branches[0.0] = Branch(n, self.root)
        self.removed_branches[self.sequence_length] = Branch()
        self.start_tree = self.get_tree_at(0.0)
        self.cut_pos = 0.0
        self.start = 0.0
        self.end = self.sequence_length

    def build_singleton_arg(self, n: Node) -> None:
        """Build an ARG containing a single sample node *n*."""
        self.add_sample(n)
        branch = Branch(n, self.root)
        r0 = Recombination(set(), {branch})
        r0.set_pos(0.0)
        self.recombinations[0.0] = r0
        for pos in list(self.mutation_sites.keys()):
            self.mutation_branches[pos] = {branch}

    def add_node(self, n: Node) -> None:
        if n is not self.root and n is not None:
            self.node_set.add(n)

    # ------------------------------------------------------------------
    # Grid / rate construction
    # ------------------------------------------------------------------

    def discretize(self, bin_size: float) -> None:
        """Build coordinate grid, placing breakpoints at recombinations.

        Mirrors ARG::discretize.
        """
        self.bin_size = bin_size
        self.coordinates = []
        recomb_keys = list(self.recombinations.keys())
        recomb_idx = 1  # skip the sentinel at 0

        curr_pos = 0.0
        while curr_pos < self.sequence_length:
            self.coordinates.append(curr_pos)
            next_recomb = recomb_keys[recomb_idx] if recomb_idx < len(recomb_keys) else _INT_MAX
            if next_recomb < curr_pos + bin_size:
                curr_pos = next_recomb
                recomb_idx += 1
            else:
                curr_pos = min(curr_pos + bin_size, self.sequence_length)
        self.coordinates.append(self.sequence_length)
        self.bin_num = len(self.coordinates) - 1

    def get_index(self, x: float) -> int:
        """Return index i such that coordinates[i] <= x < coordinates[i+1]."""
        import bisect
        idx = bisect.bisect_right(self.coordinates, x) - 1
        return max(0, idx)

    def compute_rhos_thetas(self, r: float, m: float) -> None:
        """Compute per-bin scaled recombination/mutation rates.

        Mirrors ARG::compute_rhos_thetas(double r, double m).
        """
        n = len(self.coordinates) - 1
        self.rhos = []
        self.thetas = []
        for i in range(n):
            span = self.coordinates[i + 1] - self.coordinates[i]
            self.rhos.append(r * span)
            self.thetas.append(m * span)

    # ------------------------------------------------------------------
    # Tree access
    # ------------------------------------------------------------------

    def get_tree_at(self, x: float) -> Tree:
        """Return the marginal tree at position *x*.

        Replays all Recombination records with pos <= x.
        """
        tree = Tree()
        for pos, r in self.recombinations.items():
            if pos <= x:
                tree.forward_update(r)
            else:
                break
        return tree

    def get_query_node_at(self, x: float) -> Optional[Node]:
        """Return the query node (lower_node of the removed branch at x)."""
        idx = self.removed_branches.bisect_right(x) - 1
        if idx < 0:
            return None
        key = self.removed_branches.keys()[idx]
        b = self.removed_branches[key]
        return b.lower_node if b else None

    # ------------------------------------------------------------------
    # MCMC: remove
    # ------------------------------------------------------------------

    def remove(self, cut_point: Tuple[float, Branch, float]) -> None:
        """Remove a lineage from the ARG.

        cut_point = (pos, center_branch, cut_time).

        After this call:
        - self.removed_branches maps positions to the removed branch.
        - self.joining_branches maps positions to the "joining" branch.
        - self.start / self.end delimit the affected genomic region.
        - self.start_tree is the marginal tree at start (minus the cut).
        """
        pos, center_branch, t = cut_point
        self.cut_time = t
        self.cut_node = Node(time=t)
        self.cut_node.index = -2

        self.removed_branches = SortedDict()
        self.joining_branches = SortedDict()

        forward_tree = self.cut_tree.copy()
        backward_tree = self.cut_tree.copy()

        # ---- forward pass: trace the removed branch to the right ----
        f_it_idx = self.recombinations.bisect_right(pos)
        keys = list(self.recombinations.keys())

        prev_joining = Branch()
        prev_removed = center_branch
        next_removed = center_branch

        while next_removed:
            if f_it_idx >= len(keys):
                break
            r_pos = keys[f_it_idx]
            r = self.recombinations[r_pos]
            prev_joining = forward_tree.find_joining_branch(prev_removed)
            forward_tree.forward_update(r)
            next_removed = r.trace_forward(t, prev_removed)
            if next_removed and next_removed.upper_node is self.root:
                next_removed = Branch()
            next_joining = forward_tree.find_joining_branch(next_removed)
            r.remove(prev_removed, next_removed, prev_joining, next_joining, self.cut_node)
            store_pos = min(r_pos, self.sequence_length)
            self.removed_branches[store_pos] = next_removed
            self.joining_branches[store_pos] = next_joining
            f_it_idx += 1
            prev_removed = next_removed

        # ---- backward pass: trace the removed branch to the left ----
        b_it_idx = self.recombinations.bisect_right(pos) - 1
        next_removed = center_branch
        prev_removed = center_branch

        while prev_removed:
            if b_it_idx < 0:
                break
            r_pos = keys[b_it_idx]
            r = self.recombinations[r_pos]
            self.removed_branches[r_pos] = prev_removed
            next_joining = backward_tree.find_joining_branch(next_removed)
            self.joining_branches[r_pos] = next_joining
            backward_tree.backward_update(r)
            prev_removed = r.trace_backward(t, next_removed)
            if prev_removed and prev_removed.upper_node is self.root:
                prev_removed = Branch()
            if not prev_removed:
                backward_tree.forward_update(r)
            prev_joining = backward_tree.find_joining_branch(prev_removed)
            r.remove(prev_removed, next_removed, prev_joining, next_joining, self.cut_node)
            b_it_idx -= 1
            next_removed = prev_removed

        # ---- update start / end ----
        if self.removed_branches:
            self.start = self.removed_branches.keys()[0]
            self.end = self.removed_branches.keys()[-1]
        self._remove_empty_recombinations()
        self._remap_mutations()
        self.cut_tree.remove(center_branch, self.cut_node)
        backward_tree.remove(
            self.removed_branches[self.removed_branches.keys()[0]],
            self.cut_node,
        )
        self.start_tree = backward_tree
        self.end_tree = forward_tree

    # ------------------------------------------------------------------
    # MCMC: add
    # ------------------------------------------------------------------

    def add(
        self,
        new_joining_branches: SortedDict,
        added_branches: SortedDict,
    ) -> None:
        """Thread the removed lineage back in at new positions.

        Mirrors ARG::add.
        """
        join_keys = list(new_joining_branches.keys())
        add_keys = list(added_branches.keys())
        join_idx = 0
        add_idx = 0

        r_keys = list(self.recombinations.keys())
        r_start_idx = self.recombinations.bisect_left(self.start)

        prev_joining = Branch()
        next_joining = Branch()
        prev_added = Branch()
        next_added = Branch()

        r_idx = r_start_idx

        while add_idx < len(add_keys):
            add_pos = add_keys[add_idx]
            if add_pos >= self.sequence_length:
                break

            # Advance join pointer
            if join_idx < len(join_keys) and join_keys[join_idx] == add_pos:
                next_joining = new_joining_branches[join_keys[join_idx]]
                join_idx += 1

            next_added = added_branches[add_pos]

            # Advance r_idx past any recombinations before add_pos
            while r_idx < len(r_keys) and r_keys[r_idx] < add_pos:
                r_idx += 1

            # Is there an existing recombination at add_pos?
            if r_idx < len(r_keys) and r_keys[r_idx] == add_pos:
                r = self.recombinations[r_keys[r_idx]]
                r_idx += 1
                r.add(prev_added, next_added, prev_joining, next_joining, self.cut_node)
            else:
                self._new_recombination(
                    add_pos,
                    prev_added, prev_joining,
                    next_added, next_joining,
                )

            prev_joining = next_joining
            prev_added = next_added
            add_idx += 1

        self._remove_empty_recombinations()
        self._impute(new_joining_branches, added_branches)
        # Update start_tree
        first_join = new_joining_branches[new_joining_branches.keys()[0]]
        first_added = added_branches[added_branches.keys()[0]]
        self.start_tree.add(first_added, first_join, self.cut_node)

    # ------------------------------------------------------------------
    # Recombination time sampling
    # ------------------------------------------------------------------

    def approx_sample_recombinations(self) -> None:
        """Sample start_times and finalize derived fields for all recombinations.

        Mirrors ARG::approx_sample_recombinations / RSP_smc::approx_sample_recombination.
        Sets source_branch, start_time, then calls _find_target_branch and
        _find_recomb_info so that merging_branch (used by BSP) is valid.
        """
        for pos, r in self.recombinations.items():
            if pos == 0 or pos >= self.sequence_length:
                continue
            if r.start_time > 0:
                continue
            if not r.deleted_branches:
                continue
            if r.deleted_node is None or r.inserted_node is None:
                continue

            # Find source candidates: deleted branches ending at deleted_node
            # whose induced recombined branch is in inserted_branches.
            source_candidates = []
            for b in r.deleted_branches:
                if (b.upper_node is r.deleted_node
                        and b.lower_node.time < r.inserted_node.time):
                    candidate = Branch(b.lower_node, r.inserted_node)
                    if r.create(candidate):
                        source_candidates.append(b)

            if len(source_candidates) == 1:
                r.source_branch = source_candidates[0]
                lb = max(self.cut_time, r.source_branch.lower_node.time)
                ub = min(r.source_branch.upper_node.time, r.inserted_node.time)
                r.start_time = _choose_time(lb, ub)
            elif len(source_candidates) == 2:
                lb1 = max(self.cut_time, source_candidates[0].lower_node.time)
                lb2 = max(self.cut_time, source_candidates[1].lower_node.time)
                ub1 = min(source_candidates[0].upper_node.time, r.inserted_node.time)
                ub2 = min(source_candidates[1].upper_node.time, r.inserted_node.time)
                q = (ub1 - lb1) / (ub1 + ub2 - lb1 - lb2) if (ub1 + ub2 - lb1 - lb2) > 0 else 0.5
                if 0.5 <= q:
                    r.source_branch = source_candidates[0]
                    r.start_time = _choose_time(lb1, ub1)
                else:
                    r.source_branch = source_candidates[1]
                    r.start_time = _choose_time(lb2, ub2)
            else:
                continue  # skip if no valid source found

            # Handle degenerate case: deleted and inserted nodes at same time
            if r.deleted_node.time == r.inserted_node.time:
                r.inserted_node.time = math.nextafter(r.inserted_node.time, math.inf)

            r._find_target_branch()
            r._find_recomb_info()

            ub = min(r.deleted_node.time, r.inserted_node.time)
            if r.start_time >= ub:
                r.start_time = math.nextafter(ub, -math.inf)

    # ------------------------------------------------------------------
    # Cut-point sampling (used by Threader / Sampler)
    # ------------------------------------------------------------------

    def sample_internal_cut(self) -> Tuple[float, Branch, float]:
        """Sample a random (pos, branch, cut_time) for the next MCMC step.

        Mirrors ARG::sample_internal_cut.
        """
        if self.end >= self.sequence_length - 0.1:
            self.cut_pos = 0.0
            self.cut_tree = self.get_tree_at(0.0)
        else:
            self.cut_tree = self.end_tree.copy()
            self.cut_pos = self.end

        b, t = self._tree_sample_cut_point(self.cut_tree)
        # Avoid sampling exactly at node boundaries
        max_tries = 20
        for _ in range(max_tries):
            if t != b.lower_node.time and t != b.upper_node.time:
                break
            b, t = self._tree_sample_cut_point(self.cut_tree)

        return (self.cut_pos, b, t)

    def _tree_sample_cut_point(self, tree: Tree) -> Tuple[Branch, float]:
        """Sample a (branch, time) pair uniformly over the tree.

        Mirrors Tree::sample_cut_point.
        """
        # Find root time (max non-inf parent time)
        max_time = 0.0
        for child, parent in tree.parents.items():
            if not math.isinf(parent.time) and parent.time > max_time:
                max_time = parent.time

        cut_time = self._random() * max_time
        candidates = [
            Branch(child, parent)
            for child, parent in tree.parents.items()
            if not math.isinf(parent.time)
            and parent.time > cut_time and child.time <= cut_time
        ]
        if not candidates:
            # Fallback: pick any branch
            candidates = [Branch(c, p) for c, p in tree.parents.items()
                          if not math.isinf(p.time)]
        if not candidates:
            raise RuntimeError("No valid branches in tree for cut point sampling")

        idx = int(math.floor(len(candidates) * self._random()))
        idx = min(len(candidates) - 1, idx)
        return candidates[idx], cut_time

    # ------------------------------------------------------------------
    # Check-points (for BSP / TSP sanity checks)
    # ------------------------------------------------------------------

    def get_check_points(self) -> Set[float]:
        """Return the set of recombination positions that need sanity checks."""
        if not self.removed_branches:
            return set()
        start_pos = self.removed_branches.keys()[0]
        end_pos = self.removed_branches.keys()[-1]

        deleted_nodes: Dict[Optional[Node], float] = {}
        node_spans = []

        for pos in self.recombinations.irange(start_pos, end_pos):
            r = self.recombinations[pos]
            if r.deleted_node is not None:
                deleted_nodes[r.deleted_node] = pos
            ins = r.inserted_node
            if ins is not None and ins in deleted_nodes and ins is not self.root:
                node_spans.append((ins, deleted_nodes[ins], pos))
                del deleted_nodes[ins]

        check_points: Set[float] = set()
        for n, x, y in node_spans:
            if not self._check_disjoint_nodes(x, y):
                check_points.add(y)
        return check_points

    def _check_disjoint_nodes(self, x: float, y: float) -> bool:
        r_start = self.recombinations[x]
        if r_start.deleted_node is None:
            return False
        t = r_start.deleted_node.time
        b = r_start.merging_branch
        for pos in self.recombinations.irange(x, y - 1e-15):
            r = self.recombinations[pos]
            b = r.trace_forward(t, b)
            if not b:
                return False
        r_end = self.recombinations.get(y)
        if r_end is None:
            return False
        return b == r_end.target_branch

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_remove_info(self) -> None:
        """Reset MCMC working state."""
        self.removed_branches = SortedDict()
        self.joining_branches = SortedDict()
        self.cut_node = None

    def count_flipping(self) -> int:
        count = 0
        for pos, branches in self.mutation_branches.items():
            if len(branches) > 1:
                # Check if root branch is in the set
                for b in branches:
                    if b.upper_node is self.root:
                        count += 1
                        break
        return count

    def count_incompatibility(self) -> int:
        count = 0
        for pos, branches in self.mutation_branches.items():
            if len(branches) > 1:
                has_root = any(b.upper_node is self.root for b in branches)
                if has_root:
                    if len(branches) > 2:
                        count += 1
                else:
                    count += 1
        return count

    def num_unmapped(self) -> int:
        return self.count_incompatibility()

    def get_arg_length(self) -> float:
        """Total ARG length (sum of branch_length * genomic_span)."""
        tree = self.get_tree_at(0.0)
        prev_pos = 0.0
        total = 0.0
        r_iter = iter(self.recombinations.items())
        # skip sentinel at 0
        next(r_iter)
        tree_len = tree.length()
        for r_pos, r in r_iter:
            next_pos = min(r_pos, self.sequence_length)
            total += tree_len * (next_pos - prev_pos)
            if r_pos >= self.sequence_length:
                break
            tree.forward_update(r)
            tree_len = tree.length()
            prev_pos = next_pos
        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_empty_recombinations(self) -> None:
        """Remove recombinations in [start, end] with no branch changes."""
        to_delete = []
        for pos in self.recombinations.irange(self.start, self.end):
            r = self.recombinations[pos]
            if (not r.deleted_branches and not r.inserted_branches
                    and pos < self.sequence_length):
                to_delete.append(pos)
        for pos in to_delete:
            del self.recombinations[pos]

    def _new_recombination(
        self,
        pos: float,
        prev_added: Branch,
        prev_joining: Branch,
        next_added: Branch,
        next_joining: Branch,
    ) -> None:
        """Create a new Recombination record at *pos*.

        Mirrors ARG::new_recombination.
        """
        deleted = set()
        inserted = set()
        # Branches to delete (exist before pos, not after)
        def _safe_add(s, b):
            if b.lower_node is not None and b.upper_node is not None:
                s.add(b)

        _safe_add(deleted, prev_added)
        _safe_add(deleted, Branch(prev_joining.lower_node, prev_added.upper_node))
        _safe_add(deleted, Branch(prev_added.upper_node, prev_joining.upper_node))
        _safe_add(deleted, next_joining)

        _safe_add(inserted, next_added)
        _safe_add(inserted, Branch(next_joining.lower_node, next_added.upper_node))
        _safe_add(inserted, Branch(next_added.upper_node, next_joining.upper_node))
        _safe_add(inserted, prev_joining)

        r = Recombination(deleted, inserted)
        r.set_pos(pos)
        self.recombinations[pos] = r

    def _remap_mutations(self) -> None:
        """Update mutation_branches after removing a lineage.

        Mirrors ARG::remap_mutations.
        """
        if not self.joining_branches or not self.removed_branches:
            return

        x = self.joining_branches.keys()[0]
        y = self.joining_branches.keys()[-1]

        join_idx = 0
        remove_idx = 0
        join_keys = list(self.joining_branches.keys())
        remove_keys = list(self.removed_branches.keys())

        joining_branch = Branch()
        removed_branch = Branch()

        mut_keys = list(k for k in self.mutation_branches.keys() if x <= k < y)

        for m in mut_keys:
            # Advance join pointer
            while join_idx < len(join_keys) - 1 and join_keys[join_idx + 1] <= m:
                join_idx += 1
            while remove_idx < len(remove_keys) - 1 and remove_keys[remove_idx + 1] <= m:
                remove_idx += 1

            joining_branch = self.joining_branches[join_keys[join_idx]]
            removed_branch = self.removed_branches[remove_keys[remove_idx]]

            if not joining_branch or not removed_branch:
                continue

            joining_node = removed_branch.upper_node
            if joining_node is None:
                continue

            lower_branch = Branch(joining_branch.lower_node, joining_node)
            upper_branch = Branch(joining_node, joining_branch.upper_node)

            branches = self.mutation_branches.get(m, set())
            branches.discard(removed_branch)
            branches.discard(lower_branch)
            branches.discard(upper_branch)

            sl = lower_branch.lower_node.get_state(m) if lower_branch.lower_node else 0
            su = upper_branch.upper_node.get_state(m) if upper_branch.upper_node else 0
            if sl != su:
                branches.add(joining_branch)
            self.mutation_branches[m] = branches

    def _impute(
        self,
        new_joining_branches: SortedDict,
        added_branches: SortedDict,
    ) -> None:
        """Impute mutation states for newly threaded node.

        Mirrors ARG::impute.
        """
        add_keys = list(added_branches.keys())
        join_keys = list(new_joining_branches.keys())
        join_idx = 0

        joining_branch = Branch()
        added_branch = Branch()

        for i, add_pos in enumerate(add_keys[:-1]):
            next_add_pos = add_keys[i + 1]
            added_branch = added_branches[add_pos]

            if join_idx < len(join_keys) and join_keys[join_idx] == add_pos:
                joining_branch = new_joining_branches[join_keys[join_idx]]
                join_idx += 1

            for m in self.mutation_sites.irange(add_pos, next_add_pos - 1e-15):
                if m == _INT_MAX:
                    break
                self._map_mutation_branch(m, joining_branch, added_branch)

    def _map_mutation_branch(
        self,
        x: float,
        joining_branch: Branch,
        added_branch: Branch,
    ) -> None:
        """Update mutation_branches[x] after threading added_branch.

        Mirrors ARG::map_mutation(double, Branch, Branch).
        """
        if joining_branch.is_null() or added_branch.is_null():
            return

        sl = joining_branch.lower_node.get_state(x)
        su = joining_branch.upper_node.get_state(x)
        s0 = added_branch.lower_node.get_state(x)

        sm = 1 if (sl + su + s0 > 1) else 0
        added_branch.upper_node.write_state(x, sm)

        branches = self.mutation_branches.get(x, set())
        if sl != su:
            branches.discard(joining_branch)
        if sm != sl:
            branches.add(Branch(joining_branch.lower_node, added_branch.upper_node))
        if sm != su:
            branches.add(Branch(added_branch.upper_node, joining_branch.upper_node))
        if sm != s0:
            branches.add(added_branch)
        self.mutation_branches[x] = branches

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_recombs = len(self.recombinations) - 2  # exclude sentinels
        return (
            f"ARG(samples={len(self.sample_nodes)}, "
            f"recombs={n_recombs}, "
            f"length={self.sequence_length})"
        )
