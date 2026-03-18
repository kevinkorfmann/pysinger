# Data structures

All core types live in `pysinger.data`. They form the ARG representation that the HMM and MCMC modules operate on.

## Node

```{eval-rst}
.. autoclass:: pysinger.data.node.Node
   :members:
```

A vertex in the ARG. Leaf nodes (samples) have `time = 0`; internal nodes have `time > 0` (in coalescent units); the root sentinel has `time = inf, index = -1`.

**Identity semantics.** `Node` uses Python object identity for equality and hashing (`__eq__` is `self is other`, `__hash__` returns `id(self)`). This mirrors the C++ implementation where nodes are compared by pointer. It means two `Node` objects with identical fields are still considered different -- what matters is the object itself, not its contents.

**Allele states.** Each node stores a `SortedDict` mapping genomic position $\to$ allele state (0 = ancestral, 1 = derived). A sentinel at position $-1$ with state 0 ensures lookups before the first real site always return 0. `get_state(pos)` returns an exact-match lookup (mutations are point events, not intervals).

## Branch

```{eval-rst}
.. autoclass:: pysinger.data.branch.Branch
   :members:
```

A directed edge `(lower_node, upper_node)` with `lower_node.time <= upper_node.time`. Branches are **immutable** after construction (`__setattr__` raises). Equality and hashing are based on **node identity** (not node field values), matching the C++ `branch_hash` struct.

The **null branch** (`Branch()`) has both nodes as `None` and is used as a sentinel throughout the codebase (e.g., to mark the end of a removed lineage). `bool(branch)` returns `False` for null branches.

## Tree

```{eval-rst}
.. autoclass:: pysinger.data.tree.Tree
   :members:
```

A marginal coalescent tree at one genomic position. Stored as two dicts:
- `parents: Dict[Node, Node]` -- child $\to$ parent
- `children: Dict[Node, Set[Node]]` -- parent $\to$ children

The topology is updated by applying `Recombination` records:
- `forward_update(r)`: delete `r.deleted_branches`, insert `r.inserted_branches`
- `backward_update(r)`: reverse of `forward_update`

**`find_joining_branch(removed_branch)`** is a key method used during MCMC. Given the branch being removed, it returns `Branch(sibling, grandparent)` -- the branch that "takes over" when the removed branch's coalescence node is pruned.

## Recombination

```{eval-rst}
.. autoclass:: pysinger.data.recombination.Recombination
   :members:
```

A topology change at a single genomic position. Stores:
- `deleted_branches`: branches that exist *before* (left of) this position
- `inserted_branches`: branches that exist *after* (right of) this position

From these, it derives several named branches used by the HMM:

| Branch | Meaning |
|--------|---------|
| `source_branch` | The lineage that recombines (detaches) |
| `target_branch` | The lineage it re-coalesces with |
| `merging_branch` | Sibling-to-grandparent branch after the source node is removed |
| `recombined_branch` | Source lineage below the recombination height, re-attached |
| `lower_transfer_branch` | Target branch below the new coalescence node |
| `upper_transfer_branch` | Target branch above the new coalescence node |
| `start_time` | Height at which recombination occurs |

These derived fields are computed by `_find_nodes()`, `_find_target_branch()`, and `_find_recomb_info()`, and are essential for the BSP/TSP transfer steps.

**`trace_forward(t, branch)`** and **`trace_backward(t, branch)`** map a branch at time $t$ across the topology change, used by `ARG.remove()` to trace a lineage through recombination events.

## Interval and IntervalInfo

```{eval-rst}
.. autoclass:: pysinger.data.interval.Interval
   :members:
.. autoclass:: pysinger.data.interval.IntervalInfo
   :members:
```

**Interval** represents a `(branch, [lb, ub])` cell in the HMM state space. The BSP maintains a list of `Interval` objects as its current states. Key fields:
- `weight`: coalescent probability mass $F(u_b) - F(l_b)$ (set by `CoalescentCalculator`)
- `time`: representative time point (exponential median via `fill_time()`)
- `source_weights` / `source_intervals`: traceback pointers from BSP transfer

**`fill_time()`** computes the exponential median of $[lb, ub]$: the time $t$ where $F_{\text{Exp}(1)}(t) = \frac{1}{2}[F_{\text{Exp}(1)}(lb) + F_{\text{Exp}(1)}(ub)]$. For the standard exponential, $F(t) = 1 - e^{-t}$, giving:

$$
t = -\log\!\Bigl(1 - \tfrac{1}{2}\bigl[(1-e^{-lb}) + (1-e^{-ub})\bigr]\Bigr)
$$

**IntervalInfo** is a lightweight hashable key `(branch, lb, ub)` used as dict keys in `BSP.transfer()` to accumulate probability mass from multiple source intervals into a single target.

## ARG

```{eval-rst}
.. autoclass:: pysinger.data.arg.ARG
   :members:
```

The central data structure. An ARG is encoded as a `SortedDict[position -> Recombination]` with sentinels at position 0 and $\infty$. Retrieving the marginal tree at position $x$ is done by replaying all recombinations up to $x$.

### Key operations

**`remove(cut_point)`**: Extract a lineage from the ARG. Traces the cut branch forward and backward through recombination records, building `removed_branches` and `joining_branches` maps. Modifies the recombination records in-place (via `Recombination.remove()`), re-maps mutations, and updates the `start_tree` / `end_tree`.

**`add(joining_branches, added_branches)`**: Thread a lineage back in. Walks through the added positions, either modifying existing recombination records (via `Recombination.add()`) or creating new ones. Imputes mutation states for the newly threaded nodes via majority-rule.

**`get_arg_length()`**: Total ARG branch length $= \sum_{\text{trees}} \sum_{\text{branches } b} (t_{\text{upper}}(b) - t_{\text{lower}}(b)) \times \text{genomic span}$. Used by the rescaling step.

**`approx_sample_recombinations()`**: After threading, assigns `start_time` and derived fields to all recombination records by finding the source branch and sampling a recombination time in $[\max(t_c, t_{\text{lower}}), \min(t_{\text{upper}}, t_{\text{inserted}})]$.

**`sample_internal_cut()`**: Sample a random `(position, branch, time)` triple for the next MCMC move. The time is drawn uniformly over the tree height; the branch is the one spanning that time.
