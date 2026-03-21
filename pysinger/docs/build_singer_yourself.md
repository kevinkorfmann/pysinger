# Build SINGER yourself

This chapter walks you through implementing SINGER from scratch. Each step tells you exactly which class or function to write next, shows you the Python code, explains the math in plain language, and connects it all together. By the end you will have a working Bayesian ARG sampler.

The guide follows the dependency order: data structures first, then the math helpers, then the two HMMs, then the MCMC loop, and finally I/O.

---

## Step 1 — The Node class

**Goal.** Represent a single vertex in the ARG.

Every genealogical tree is made of nodes connected by branches. A `Node` is a vertex — it could be a present-day sample (a haplotype you observed) or an ancestor where two lineages coalesced. Each node needs three pieces of information:

- **`time`** — when this ancestor lived, measured in coalescent units (where 1 unit = $N_e$ generations). Present-day samples have `time = 0`. The further back in time, the larger the number. You also need a special **root sentinel** with `time = inf` — every tree in the ARG hangs off this sentinel so you never have to special-case "which node is the root?"

- **`index`** — an integer label. Samples get `0, 1, …, n-1`. Internal nodes get larger numbers. The root sentinel gets `-1`.

- **`mutation_sites`** — a sorted map from genomic position to allele state. If a node carries the derived allele at position 500, then `mutation_sites[500] = 1`. We use a `SortedDict` because we need efficient range queries later. A **sentinel entry at position `-1` with state `0`** ensures that any lookup before the first real site returns 0 (ancestral) without special-casing.

```python
from dataclasses import dataclass, field
from sortedcontainers import SortedDict

@dataclass
class Node:
    time: float
    index: int = 0
    mutation_sites: SortedDict = field(
        default_factory=lambda: SortedDict({-1: 0})
    )

    def add_mutation(self, pos: float) -> None:
        """Record that this node carries the derived allele at pos."""
        self.mutation_sites[pos] = 1

    def get_state(self, pos: float) -> float:
        """Return allele state at pos (exact match only, else 0).

        Mutations are point events: a mutation at position 100 does NOT
        carry its state to position 101.  This is just a dict lookup.
        """
        return self.mutation_sites.get(pos, 0)

    def write_state(self, pos: float, state: float) -> None:
        """Set the allele state; state==0 removes the entry."""
        if state == 0:
            if pos != -1:  # never remove the sentinel
                self.mutation_sites.pop(pos, None)
        elif state == 1:
            self.mutation_sites[pos] = state

    # CRITICAL: identity-based equality and hashing
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
```

**Why identity semantics?** In the ARG, two different coalescence events can happen at the same time. If you compared nodes by their `time` and `index` fields, you couldn't tell them apart. By using `self is other` for equality and `id(self)` for hashing, every `Node` object is unique — even if two nodes have identical fields. This mirrors the C++ implementation where nodes are compared by pointer address.

> **Code:** `pysinger/data/node.py`

---

## Step 2 — The Branch class

**Goal.** Represent a directed edge between two nodes.

A branch connects a child node (`lower_node`, closer to the present) to a parent node (`upper_node`, further in the past). The invariant is `lower_node.time ≤ upper_node.time`.

```python
class Branch:
    __slots__ = ("lower_node", "upper_node")

    def __init__(self, lower_node=None, upper_node=None):
        object.__setattr__(self, "lower_node", lower_node)
        object.__setattr__(self, "upper_node", upper_node)

    def __setattr__(self, name, value):
        raise AttributeError("Branch is immutable")

    def __eq__(self, other):
        if not isinstance(other, Branch):
            return False
        return (self.lower_node is other.lower_node
                and self.upper_node is other.upper_node)

    def __hash__(self):
        return hash((id(self.lower_node), id(self.upper_node)))

    @property
    def length(self):
        if self.lower_node is None or self.upper_node is None:
            return float("inf")
        return self.upper_node.time - self.lower_node.time

    def is_null(self):
        return self.lower_node is None and self.upper_node is None

    def __bool__(self):
        return not self.is_null()
```

Three important design decisions:

1. **Immutability.** After construction, you cannot change which nodes a branch connects. We enforce this by overriding `__setattr__` to raise an error, and using `object.__setattr__` inside `__init__` to bypass our own guard. This is necessary because branches are used as dictionary keys and set members — if they could change, the hash would become invalid.

2. **Identity-based hashing.** Just like nodes, branches are compared by the *identity* of their node objects, not by the values of their fields. `__hash__` uses `id(self.lower_node)` and `id(self.upper_node)`. This means `Branch(nodeA, nodeB) == Branch(nodeA, nodeB)` is `True` only if `nodeA` and `nodeB` are the exact same objects in memory.

3. **Null branch.** `Branch()` with both nodes as `None` is used as a sentinel throughout the codebase — for example, to mark the end of a lineage that has been traced to its boundary. `bool(branch)` returns `False` for null branches, so you can write `if branch:` to check.

> **Code:** `pysinger/data/branch.py`

---

## Step 3 — The Tree class

**Goal.** Represent a marginal coalescent tree at a single genomic position.

An ARG is a sequence of trees along the genome. At each position, there is one tree that describes how the samples are related. We store each tree as two dictionaries:

```python
class Tree:
    def __init__(self):
        self.parents = {}    # child Node → parent Node
        self.children = {}   # parent Node → set of child Nodes
```

These two dicts are redundant (one could be derived from the other), but keeping both makes lookups fast in both directions.

### Adding and removing branches

Every topology change is built from two atomic operations:

```python
def insert_branch(self, branch):
    """Add a branch to the tree."""
    ln, un = branch.lower_node, branch.upper_node
    self.parents[ln] = un
    if un not in self.children:
        self.children[un] = set()
    self.children[un].add(ln)

def delete_branch(self, branch):
    """Remove a branch from the tree."""
    ln, un = branch.lower_node, branch.upper_node
    self.parents.pop(ln, None)
    if un in self.children:
        self.children[un].discard(ln)
        if not self.children[un]:
            del self.children[un]
```

### Moving the tree along the genome

As you walk along the genome, the tree changes at recombination breakpoints. Each change is recorded as a `Recombination` object (Step 4) that lists which branches to delete and which to insert:

```python
def forward_update(self, r):
    """Apply recombination r, moving right along the genome."""
    for b in r.deleted_branches:
        self.delete_branch(b)
    for b in r.inserted_branches:
        self.insert_branch(b)

def backward_update(self, r):
    """Undo recombination r, moving left along the genome."""
    for b in r.inserted_branches:
        self.delete_branch(b)
    for b in r.deleted_branches:
        self.insert_branch(b)
```

### Finding the joining branch (key for MCMC)

When you remove a lineage from a tree (the core MCMC move), you need to know what happens to the remaining tree. If you remove a branch, its coalescence node disappears, and the sibling gets connected directly to the grandparent. `find_joining_branch` computes this:

```python
def find_sibling(self, n):
    """Return the other child of n's parent."""
    p = self.parents[n]
    for c in self.children[p]:
        if c is not n:
            return c
    return None

def find_joining_branch(self, removed_branch):
    """After removing this branch, what branch 'takes over'?

    Returns Branch(sibling, grandparent).
    """
    if removed_branch.is_null():
        return Branch()
    sibling = self.find_sibling(removed_branch.lower_node)
    if sibling is None:
        return Branch()
    grandparent = self.parents.get(removed_branch.upper_node)
    if grandparent is None:
        return Branch(sibling, removed_branch.upper_node)
    return Branch(sibling, grandparent)
```

Think of it this way: in a binary tree, every internal node has exactly two children. If you remove one child, the internal node becomes unnecessary — you can "short-circuit" the sibling directly to the grandparent. This is the reverse of a subtree-prune-and-regraft (SPR) operation.

### Total branch length

```python
def length(self):
    """Sum of all branch lengths (excluding root sentinel branches)."""
    total = 0.0
    for child, parent in self.parents.items():
        if parent.index != -1:  # skip root sentinel
            total += parent.time - child.time
    return total
```

This is used later by the rescaling step to calibrate coalescence times to the mutation rate.

> **Code:** `pysinger/data/tree.py`

---

## Step 4 — The Recombination class

**Goal.** Record a single topology change at a genomic breakpoint.

Under the Sequentially Markov Coalescent (SMC), adjacent marginal trees differ by exactly one SPR operation. At position `pos`, one lineage detaches from its current parent and re-attaches somewhere else. A `Recombination` stores the "before" and "after" as two sets of branches:

```python
class Recombination:
    def __init__(self, deleted_branches=None, inserted_branches=None):
        self.pos = 0.0
        self.deleted_branches = set(deleted_branches or [])   # exist BEFORE pos
        self.inserted_branches = set(inserted_branches or []) # exist AFTER pos

        # Derived fields (computed below)
        self.deleted_node = None    # old coalescence node being removed
        self.inserted_node = None   # new coalescence node being created
        self.source_branch = Branch()      # lineage that recombines
        self.target_branch = Branch()      # lineage it re-coalesces with
        self.merging_branch = Branch()     # sibling→grandparent after removal
        self.recombined_branch = Branch()  # source below recombination height
        self.lower_transfer_branch = Branch()  # target below new node
        self.upper_transfer_branch = Branch()  # target above new node
        self.start_time = 0.0              # height of recombination

        if self.deleted_branches or self.inserted_branches:
            self._simplify_branches()
            self._find_nodes()
```

### Deriving the named branches

The SPR operation involves several named branches that the HMMs need. Here is how to identify them from the deleted/inserted sets:

**`_find_nodes()`** — identify which coalescence nodes were destroyed and created:

```python
def _find_nodes(self):
    """The deleted_node appears as an upper_node in deleted but not inserted.
    The inserted_node appears as an upper_node in inserted but not deleted."""
    prev_upper = {b.upper_node for b in self.deleted_branches}
    next_upper = {b.upper_node for b in self.inserted_branches}
    for n in prev_upper:
        if n not in next_upper:
            self.deleted_node = n
    for n in next_upper:
        if n not in prev_upper:
            self.inserted_node = n
```

**`_find_target_branch()`** — find the branch that was split by the new coalescence node:

```python
def _find_target_branch(self):
    """The target branch is the deleted branch (other than source) that gets
    split into two inserted branches at inserted_node.time."""
    t = self.inserted_node.time
    for b in self.deleted_branches:
        if b == self.source_branch:
            continue
        if b.lower_node.time > t or b.upper_node.time < t:
            continue
        lower = Branch(b.lower_node, self.inserted_node)
        upper = Branch(self.inserted_node, b.upper_node)
        if lower in self.inserted_branches and upper in self.inserted_branches:
            self.target_branch = b
            return
```

**`_find_recomb_info()`** — compute the remaining named branches:

```python
def _find_recomb_info(self):
    dn = self.deleted_node
    l, u = None, None
    for b in self.deleted_branches:
        if b == self.source_branch:
            continue
        if b.upper_node is dn:
            l = self.inserted_node if b == self.target_branch else b.lower_node
        elif b.lower_node is dn:
            u = self.inserted_node if b == self.target_branch else b.upper_node

    self.merging_branch = Branch(l, u)
    self.recombined_branch = Branch(
        self.source_branch.lower_node, self.inserted_node
    )
    # Transfer branches: how the target branch splits around inserted_node
    candidate_lower = Branch(self.target_branch.lower_node, self.inserted_node)
    self.lower_transfer_branch = (
        candidate_lower if candidate_lower in self.inserted_branches
        else self.merging_branch
    )
    candidate_upper = Branch(self.inserted_node, self.target_branch.upper_node)
    self.upper_transfer_branch = (
        candidate_upper if candidate_upper in self.inserted_branches
        else self.merging_branch
    )
```

To visualise what these branches mean, consider an SPR where lineage A detaches from its parent P and re-attaches to branch B at a new node Q:

```
BEFORE (deleted):               AFTER (inserted):
    P                               Q
   / \                             / \
  A   S                           A   B_lower
      |                               |
      ...                            ...
  B = (B_lower, B_upper)        P removed; S connects to grandparent

source_branch  = (A, P)         recombined_branch = (A, Q)
target_branch  = (B_lower, B_upper)
merging_branch = (S, grandparent)
lower_transfer = (B_lower, Q)
upper_transfer = (Q, B_upper)
```

### Tracing a lineage through a recombination

The MCMC needs to follow a lineage as it passes through recombination breakpoints. `trace_forward` answers: "if I was on `branch` at time `t` before this recombination, what branch am I on after?"

```python
def trace_forward(self, t, branch):
    if not self.affect(branch):
        return branch  # this branch is not affected
    if branch == self.source_branch:
        if t >= self.start_time:
            return Branch()  # lineage was cut above the recombination
        return self.recombined_branch
    if branch == self.target_branch:
        if t > self.inserted_node.time:
            return self.upper_transfer_branch
        return self.lower_transfer_branch
    return self.merging_branch  # was sibling/parent of source
```

### Simplifying

After any modification, branches that appear in *both* sets cancel out:

```python
def _simplify_branches(self):
    common = self.deleted_branches & self.inserted_branches
    self.deleted_branches -= common
    self.inserted_branches -= common
```

> **Code:** `pysinger/data/recombination.py`

---

## Step 5 — The Interval and IntervalInfo classes

**Goal.** Represent a single cell `(branch, [lb, ub])` in the HMM state space.

The BSP and TSP are Hidden Markov Models whose states are regions of the tree. Each state is: "the new lineage coalesces on *this branch* at a time between *lb* and *ub*." An `Interval` is one such state.

```python
import math

class Interval:
    __slots__ = (
        "branch", "lb", "ub", "start_pos",
        "weight", "time",
        "source_weights", "source_intervals", "node",
    )

    def __init__(self, branch, lb, ub, start_pos):
        self.branch = branch
        self.lb = lb
        self.ub = ub
        self.start_pos = start_pos  # HMM step where this interval was created
        self.weight = 0.0           # coalescent probability mass in [lb, ub]
        self.time = 0.0             # representative time point
        self.source_weights = []    # traceback pointers (for BSP transfer)
        self.source_intervals = []
        self.node = None            # optional Node (for TSP point masses)
```

### `fill_time()` — the exponential median

Each interval needs a single representative time point for computing emission and transition probabilities. We use the **exponential median**: the time $t$ where the $\text{Exp}(1)$ CDF is halfway between $F(lb)$ and $F(ub)$.

The $\text{Exp}(1)$ CDF is $F(t) = 1 - e^{-t}$. The exponential median is:

$$
lq = 1 - e^{-lb}, \quad uq = 1 - e^{-ub}, \quad q = \frac{lq + uq}{2}, \quad t = -\log(1 - q)
$$

Why not the arithmetic midpoint? Because coalescent times are exponentially distributed. Near the bottom of a branch (close to the present), times are dense; near the top, they are sparse. The exponential median picks a representative point that respects this density — it sits where the coalescent probability mass is concentrated, not at the geometric centre.

```python
def fill_time(self):
    lb, ub = self.lb, self.ub
    if math.isinf(ub):
        self.time = lb + math.log(2)  # median of Exp shifted by lb
        return
    if abs(lb - ub) < 1e-3:
        self.time = 0.5 * (lb + ub)  # tiny interval: midpoint is fine
        return
    lq = 1.0 - math.exp(-lb)
    uq = 1.0 - math.exp(-ub)
    if uq - lq < 1e-3:
        self.time = 0.5 * (lb + ub)
    else:
        q = 0.5 * (lq + uq)
        self.time = -math.log(1.0 - q)
    self.time = max(lb, min(ub, self.time))  # clamp
```

### `full(cut_time)` — is this a full interval?

A "full" interval spans the entire branch above `cut_time`. Only full intervals participate in the BSP's recombination weight calculation — partial intervals (created during transfer steps) don't contribute to the recombination sum.

```python
def full(self, cut_time):
    lb_expected = max(cut_time, self.branch.lower_node.time)
    ub_expected = self.branch.upper_node.time
    return self.lb == lb_expected and self.ub == ub_expected
```

**`IntervalInfo`** is a lightweight hashable key `(branch, lb, ub)` used as dict keys during BSP transfer to accumulate probability mass from multiple source intervals that map to the same target region.

> **Code:** `pysinger/data/interval.py`

---

## Step 6 — The ARG class

**Goal.** The central data structure — the entire Ancestral Recombination Graph.

An ARG is stored as a **sorted map** from genomic position to `Recombination` record, with sentinel entries at position 0 and $\infty$. To get the marginal tree at any position $x$, replay all records from 0 up to $x$:

```python
from sortedcontainers import SortedDict

class ARG:
    def __init__(self, Ne=1.0, sequence_length=1.0):
        self.Ne = Ne
        self.sequence_length = sequence_length
        self.root = Node(time=math.inf, index=-1)  # root sentinel

        self.sample_nodes = set()
        self.recombinations = SortedDict()
        # Sentinel records at boundaries
        r0 = Recombination(); r0.pos = 0.0
        self.recombinations[0.0] = r0
        r_end = Recombination(); r_end.pos = float(sys.maxsize)
        self.recombinations[float(sys.maxsize)] = r_end

        # Mutation tracking
        self.mutation_sites = SortedDict()   # position → True
        self.mutation_branches = {}          # position → set of Branches

        # HMM grid
        self.coordinates = []
        self.rhos = []    # per-bin recombination rates
        self.thetas = []  # per-bin mutation rates

        # MCMC working state
        self.removed_branches = SortedDict()
        self.joining_branches = SortedDict()
        self.cut_tree = Tree()
        self.cut_time = 0.0
        self.cut_node = None
        self.start = 0.0
        self.end = sequence_length

    def get_tree_at(self, x):
        """Replay recombinations to get the marginal tree at position x."""
        tree = Tree()
        for pos, r in self.recombinations.items():
            if pos <= x:
                tree.forward_update(r)
            else:
                break
        return tree
```

### Building the simplest ARG

```python
def build_singleton_arg(self, node):
    """ARG with one sample connected to the root sentinel."""
    self.add_sample(node)
    branch = Branch(node, self.root)
    r0 = Recombination(set(), {branch})
    r0.pos = 0.0
    self.recombinations[0.0] = r0
```

### The HMM coordinate grid

The BSP and TSP operate on a discrete grid along the genome. Each grid cell is one "bin" with its own recombination rate $\rho$ and mutation rate $\theta$. Grid points are placed at regular intervals *and* at every recombination breakpoint (so that topology changes always land exactly on a grid boundary):

```python
def discretize(self, bin_size):
    self.coordinates = []
    recomb_keys = list(self.recombinations.keys())
    recomb_idx = 1  # skip sentinel at 0
    curr_pos = 0.0

    while curr_pos < self.sequence_length:
        self.coordinates.append(curr_pos)
        next_recomb = (recomb_keys[recomb_idx]
                       if recomb_idx < len(recomb_keys)
                       else float("inf"))
        if next_recomb < curr_pos + bin_size:
            curr_pos = next_recomb
            recomb_idx += 1
        else:
            curr_pos = min(curr_pos + bin_size, self.sequence_length)

    self.coordinates.append(self.sequence_length)

def compute_rhos_thetas(self, r, m):
    """Per-bin rates: rho = r * span, theta = m * span.
    r and m are already in coalescent units (multiplied by Ne)."""
    for i in range(len(self.coordinates) - 1):
        span = self.coordinates[i + 1] - self.coordinates[i]
        self.rhos.append(r * span)
        self.thetas.append(m * span)
```

### `remove(cut_point)` — extracting a lineage (MCMC)

This is the most complex operation. Given a cut point `(pos, branch, cut_time)`, it traces the lineage forward and backward through recombination records, removing it from the ARG:

```python
def remove(self, cut_point):
    pos, center_branch, t = cut_point
    self.cut_time = t
    self.cut_node = Node(time=t, index=-2)  # sentinel for the cut

    # Forward pass: trace the branch to the right
    prev_removed = center_branch
    for each recombination r to the right of pos:
        joining = tree.find_joining_branch(prev_removed)
        tree.forward_update(r)
        next_removed = r.trace_forward(t, prev_removed)
        next_joining = tree.find_joining_branch(next_removed)
        r.remove(prev_removed, next_removed, joining, next_joining, cut_node)
        self.removed_branches[r.pos] = next_removed
        self.joining_branches[r.pos] = next_joining
        prev_removed = next_removed

    # Backward pass: trace the branch to the left (symmetric)
    # ...same logic using trace_backward...
```

At each step, `r.remove(...)` updates the recombination record to reflect the topology without the removed lineage. The result is two maps: `removed_branches` (what was taken out at each position) and `joining_branches` (what filled the gap).

### `add(joining_branches, added_branches)` — threading a lineage back in

The reverse of `remove`. Walk through the `added_branches` map, updating or creating recombination records:

```python
def add(self, new_joining_branches, added_branches):
    for pos in added_branches:
        if pos is at an existing recombination:
            r.add(prev_added, next_added, prev_joining, next_joining, cut_node)
        else:
            self._new_recombination(pos, prev_added, prev_joining,
                                    next_added, next_joining)
    self._impute(new_joining_branches, added_branches)  # assign mutations
```

### `_impute()` — majority-rule mutation assignment

When you thread a new lineage, you create new coalescence nodes that need allele states. For each segregating site, look at the three nodes around the new coalescence: the lower node ($s_l$), the upper node ($s_u$), and the query node ($s_0$). The new node gets the **majority-rule state**:

$$
s_m = \begin{cases} 1 & \text{if } s_l + s_u + s_0 > 1 \\ 0 & \text{otherwise} \end{cases}
$$

This is the simplest parsimony assignment — whichever allele is in the majority wins. It is not exact (Fitch parsimony does better), but it is fast and sufficient for the MCMC.

```python
def _map_mutation_branch(self, x, joining_branch, added_branch):
    sl = joining_branch.lower_node.get_state(x)
    su = joining_branch.upper_node.get_state(x)
    s0 = added_branch.lower_node.get_state(x)
    sm = 1 if (sl + su + s0 > 1) else 0
    added_branch.upper_node.write_state(x, sm)
    # Update mutation_branches tracking...
```

### `get_arg_length()` — total branch length

Walk all marginal trees, summing `tree.length() * genomic_span` for each interval between recombinations. This gives the total amount of "evolutionary opportunity" for mutations in the ARG, which the rescaling step uses.

```python
def get_arg_length(self):
    tree = self.get_tree_at(0.0)
    prev_pos = 0.0
    total = 0.0
    for r_pos, r in self.recombinations.items():
        if r_pos == 0:
            continue
        next_pos = min(r_pos, self.sequence_length)
        total += tree.length() * (next_pos - prev_pos)
        if r_pos >= self.sequence_length:
            break
        tree.forward_update(r)
        prev_pos = next_pos
    return total
```

### `sample_internal_cut()` — choosing where to cut

For the MCMC, you need to pick a random point in the ARG to cut. Draw a random time uniformly from `[0, max_tree_height]`, then find the branch that spans that time:

```python
def sample_internal_cut(self):
    # Find max non-inf node time in the cut tree
    max_time = max(p.time for p in self.cut_tree.parents.values()
                   if not math.isinf(p.time))
    cut_time = random() * max_time

    # Find branches spanning that time
    candidates = [
        Branch(child, parent)
        for child, parent in self.cut_tree.parents.items()
        if not math.isinf(parent.time)
        and parent.time > cut_time and child.time <= cut_time
    ]
    branch = candidates[randint(0, len(candidates) - 1)]
    return (self.cut_pos, branch, cut_time)
```

> **Code:** `pysinger/data/arg.py`

---

## Step 7 — The CoalescentCalculator

**Goal.** Compute the probability that a new lineage coalesces in a given time interval.

When you add a new lineage to an existing tree, it "falls" from the present (time 0) upward into the past, and at each moment it can coalesce with any lineage alive at that time. The rate of coalescence at time $t$ equals the number of branches alive at $t$:

$$
\lambda(t) = \#\{\text{branches spanning time } t\}
$$

This rate is **piecewise constant**: it only changes when a branch starts or ends (at node times). Between those events, the coalescence process is exponential with a constant rate. The cumulative probability of coalescing by time $t$ is:

$$
F(t) = 1 - \exp\!\left(-\int_0^t \lambda(s)\,ds\right)
$$

Because $\lambda$ is piecewise constant, $F$ is **piecewise exponential** — and we can compute it exactly.

### Building the CDF

```python
class CoalescentCalculator:
    def __init__(self, cut_time):
        self.cut_time = cut_time

    def compute(self, branches):
        self._compute_rate_changes(branches)
        self._compute_rates()
        self._compute_probs_quantiles()
```

**Step 7a: Record rate changes.** For each branch, the coalescence rate goes up by 1 when the branch starts (at `max(cut_time, lower_node.time)`) and down by 1 when it ends (at `upper_node.time`):

```python
def _compute_rate_changes(self, branches):
    self._rate_changes = SortedDict()
    for b in branches:
        lb = max(self.cut_time, b.lower_node.time)
        ub = b.upper_node.time
        self._rate_changes[lb] = self._rate_changes.get(lb, 0) + 1
        self._rate_changes[ub] = self._rate_changes.get(ub, 0) - 1
```

**Step 7b: Running sum gives the piecewise constant rate.**

```python
def _compute_rates(self):
    self._rates = SortedDict()
    curr = 0
    for t, delta in self._rate_changes.items():
        curr += delta
        self._rates[t] = curr
```

Now `self._rates[t]` tells you: "from time $t$ until the next rate change, the coalescence rate is this value."

**Step 7c: Piecewise exponential CDF.** Walk pairs of adjacent rate-change times. Between $[t_k, t_{k+1})$ with rate $\lambda_k$, the survival probability decays exponentially:

$$
S(t_{k+1}) = S(t_k) \cdot e^{-\lambda_k(t_{k+1} - t_k)}
$$

In words: the probability of *not* coalescing by time $t_{k+1}$ equals the probability of surviving to $t_k$ times the probability of surviving the interval $[t_k, t_{k+1})$ with rate $\lambda_k$.

The cumulative coalescence probability grows by the mass that "coalesced" in this interval:

$$
F(t_{k+1}) = F(t_k) + S(t_k) - S(t_{k+1})
$$

```python
def _compute_probs_quantiles(self):
    self._cum_probs = SortedDict()
    rate_keys = list(self._rates.keys())
    prev_prob = 1.0  # survival probability (starts at 1)
    cum_prob = 0.0   # cumulative coalescence probability

    for i in range(len(rate_keys) - 1):
        rate = self._rates[rate_keys[i]]
        dt = rate_keys[i + 1] - rate_keys[i]
        if rate > 0:
            next_prob = prev_prob * math.exp(-rate * dt)
            cum_prob += prev_prob - next_prob
        else:
            next_prob = prev_prob
        self._cum_probs[rate_keys[i + 1]] = cum_prob
        prev_prob = next_prob

    self._cum_probs[self.min_time] = 0.0
```

### Interpolated CDF lookup

To get $F(x)$ for an arbitrary time $x$ (not just at rate-change boundaries), find the interval $[t_k, t_{k+1})$ containing $x$ and interpolate:

$$
F(x) = F(t_k) + \Delta F \cdot \frac{\text{expm1}(-\lambda_k \cdot (x - t_k))}{\text{expm1}(-\lambda_k \cdot (t_{k+1} - t_k))}
$$

This is **exact**, not an approximation — because the rate is constant within the interval, the CDF really is exponential there. The `expm1` function (`math.expm1`) computes $e^x - 1$ accurately even for small $x$, avoiding floating-point cancellation.

```python
def prob(self, x):
    # Find interval [l_key, u_key] containing x
    u_idx = self._cum_probs.bisect_right(x)
    l_idx = u_idx - 1
    l_key = self._cum_probs.keys()[l_idx]
    u_key = self._cum_probs.keys()[u_idx]

    base_prob = self._cum_probs[l_key]
    rate = self._rates.get(l_key, 0)
    if rate == 0:
        return base_prob

    delta_t = u_key - l_key
    delta_p = self._cum_probs[u_key] - base_prob
    new_delta_t = x - l_key

    denom = math.expm1(-rate * delta_t)
    if abs(denom) < 1e-15:
        new_delta_p = delta_p * new_delta_t / delta_t  # linear fallback
    else:
        new_delta_p = delta_p * math.expm1(-rate * new_delta_t) / denom

    return base_prob + new_delta_p
```

### Inverse CDF (quantile function)

Given a probability $p$, find time $t$ such that $F(t) = p$. This is the inverse of the above formula:

$$
t = t_k - \frac{1}{\lambda_k} \log\!\left(1 - \frac{p - F(t_k)}{F(t_{k+1}) - F(t_k)} \cdot (1 - e^{-\lambda_k \Delta t})\right)
$$

In plain language: find which segment of the CDF contains probability $p$, then invert the exponential formula within that segment.

```python
def quantile(self, p):
    # Find the segment where cum_prob crosses p
    idx = bisect.bisect_right(self._prob_vals, p)
    l_idx = idx - 1
    u_idx = idx

    l_time = self._prob_times[l_idx]
    l_prob = self._prob_vals[l_idx]
    delta_p = self._prob_vals[u_idx] - l_prob
    delta_t = self._prob_times[u_idx] - l_time
    rate = self._rates.get(l_time, 0)
    new_delta_p = p - l_prob

    if rate == 0:
        new_delta_t = delta_t * new_delta_p / delta_p
    else:
        frac = new_delta_p / delta_p * (1.0 - math.exp(-rate * delta_t))
        new_delta_t = -math.log(1.0 - frac) / rate

    return l_time + new_delta_t
```

### The two main methods

```python
def weight(self, lb, ub):
    """Probability of coalescence in [lb, ub]."""
    return self.prob(ub) - self.prob(lb)

def time(self, lb, ub):
    """Representative coalescence time in [lb, ub] (exponential median).

    Find t where F(t) is midway between F(lb) and F(ub)."""
    if math.isinf(ub):
        return lb + math.log(2)
    if ub - lb < 1e-3:
        return 0.5 * (lb + ub)
    lq = self.prob(lb)
    uq = self.prob(ub)
    if uq - lq < 1e-3:
        return 0.5 * (lb + ub)
    mid = 0.5 * (lq + uq)
    return max(lb, min(ub, self.quantile(mid)))
```

`weight` is used to set up the initial BSP forward probabilities. `time` is used to pick the representative time for each interval.

> **Code:** `pysinger/hmm/coalescent.py`

---

## Step 8 — Emission models

**Goal.** Compute how well the observed mutations match a proposed coalescence point.

When you propose threading a new lineage onto branch $b$ at time $t$, you split $b$ into three sub-branches:

```
                upper_node (time u)
                    |
               ℓ_upper = u - t
                    |
    query_node ----[t]---- split point
        |
   ℓ_query = t - t_q      ℓ_lower = t - l
        |                      |
                          lower_node (time l)
```

The emission probability is the **likelihood ratio**: how much more likely is the observed data with the new lineage threaded in, compared to without it?

### BinaryEmission (used by the TSP)

**No mutations in the bin (null emission):**

The Poisson probability of seeing no mutations on a branch of length $\ell$ with scaled rate $\theta$ is $e^{-\theta\ell}$. With the new lineage, the old branch $(\ell_l + \ell_u)$ is replaced by three sub-branches. The ratio is:

$$
e_{\text{null}} = \frac{e^{-\theta \ell_l} \cdot e^{-\theta \ell_u} \cdot e^{-\theta \ell_q}}{e^{-\theta(\ell_l + \ell_u)}} = e^{-\theta \ell_q}
$$

The lower and upper segments cancel in numerator and denominator (they're the same total length), leaving only the query branch's contribution. Longer query branches (coalescence further in the past) get penalised because they have more opportunity for mutations that weren't observed.

```python
class BinaryEmission:
    def null_emit(self, branch, time, theta, node):
        l_query = time - node.time
        return math.exp(-theta * l_query)
```

**With mutations (mutation emission):**

For each segregating site, compute the **majority-rule ancestral state** from the three nodes around the split:

$$
s_m = \mathbb{1}[s_l + s_u + s_0 > 1.5]
$$

where $s_l$ = state of the lower node, $s_u$ = state of the upper node, $s_0$ = state of the query node. Then count state changes on each sub-branch: $d_l = |s_m - s_l|$, $d_u = |s_m - s_u|$, $d_q = |s_m - s_0|$, and for the original unsplit branch $d_{\text{old}} = |s_l - s_u|$.

Each state change contributes a factor of $\theta/\Delta x$ (the per-site mutation rate). The emission is a product over all sites:

$$
e_{\text{mut}} = e^{-\theta \ell_q} \cdot \prod_{\text{sites}} \frac{(\theta/\Delta x)^{d_l + d_u + d_q}}{(\theta/\Delta x)^{d_{\text{old}}}}
$$

The intuition: if the new lineage "explains" a mutation better (fewer total state changes), the emission is higher. If it adds unnecessary state changes, the emission is lower.

```python
def _get_diff(self, mut_set, branch, node):
    """Returns [d_lower, d_upper, d_query, d_old]."""
    d = [0, 0, 0, 0]
    for x in mut_set:
        sl = branch.lower_node.get_state(x)
        su = branch.upper_node.get_state(x)
        s0 = node.get_state(x)
        sm = 1 if (sl + su + s0 > 1.5) else 0
        d[0] += abs(sm - sl)
        d[1] += abs(sm - su)
        d[2] += abs(sm - s0)
        d[3] += abs(sl - su)
    return d

def mut_emit(self, branch, time, theta, bin_size, mut_set, node):
    ll = time - branch.lower_node.time
    lu = branch.upper_node.time - time
    l0 = time - node.time
    diff = self._get_diff(mut_set, branch, node)
    emit = (self._calc(ll * theta, bin_size, diff[0])
          * self._calc(lu * theta, bin_size, diff[1])
          * self._calc(l0 * theta, bin_size, diff[2]))
    old = self._calc((ll + lu) * theta, bin_size, diff[3])
    return max(emit / old, 1e-20)

def _calc(self, theta, bin_size, s):
    """exp(-theta) * (theta/bin_size)^s"""
    return math.exp(-theta) * (theta / bin_size) ** s
```

### PolarEmission (used by the BSP)

Extends `BinaryEmission` with two refinements:

1. **Derived allele penalty**: if the query node differs from the majority ($d_q \geq 1$), multiply by a configurable penalty factor. This discourages placing the new lineage where it introduces unnecessary derived alleles.

2. **Root reward**: when the branch reaches the root sentinel and the majority state is ancestral while the lower node is derived, apply `ancestral_prob / (1 - ancestral_prob)`. This biases toward the prior expectation that the root carries the ancestral allele.

```python
class PolarEmission(Emission):
    def __init__(self, penalty=1.0, ancestral_prob=0.5):
        self.penalty = penalty
        self.ancestral_prob = ancestral_prob

    def null_emit(self, branch, time, theta, node):
        l0 = time - node.time
        lu = branch.upper_node.time - time
        if not math.isinf(lu):
            return math.exp(-theta * l0)
        else:
            ll = time - branch.lower_node.time
            return math.exp(-theta * (ll + l0))
```

> **Code:** `pysinger/hmm/emission.py`

---

## Step 9 — The BSP (Branch Sequence Propagator)

**Goal.** A forward HMM that answers: **which branch** should the new lineage join at each genomic position?

### The forward equation

At each position $x$, every interval $i$ carries a forward probability $\alpha_i(x)$ representing the probability that the lineage is on branch $i$'s region, given all the data seen so far. Between positions, the lineage can either *stay* on the same branch or *recombine* (jump to a new branch):

$$
\alpha_i(x+1) = \underbrace{\alpha_i(x) \cdot (1 - p_i)}_{\text{stay}} + \underbrace{R \cdot w_i}_{\text{jump here from anywhere}}
$$

where:

- $p_i = \rho \cdot (t_i - t_c) \cdot e^{-\rho(t_i - t_c)}$ is the probability that interval $i$ recombines. Longer branches (more time between the representative time $t_i$ and the cut time $t_c$) have higher recombination probability.

- $R = \sum_j p_j \cdot \alpha_j(x)$ is the total probability mass that recombines away from any interval.

- $w_i$ is the probability of *landing* on interval $i$ after recombining. It's proportional to `recomb_prob * coalescent_weight` for "full" intervals (ones spanning the entire branch), normalised to sum to 1.

```python
class BSP:
    def start(self, branches, cut_time):
        """Initialise at the left boundary."""
        self.cut_time = cut_time
        self.valid_branches = {b for b in branches
                               if b.upper_node.time > cut_time}

        self.cc = CoalescentCalculator(cut_time)
        self.cc.compute(self.valid_branches)

        self.curr_intervals = []
        initial_probs = []
        for b in sorted(self.valid_branches):
            lb = max(b.lower_node.time, cut_time)
            ub = b.upper_node.time
            iv = Interval(b, lb, ub, 0)
            self.curr_intervals.append(iv)
            initial_probs.append(self.cc.weight(lb, ub))

        self.forward_probs = [initial_probs]

    def forward(self, rho):
        """Advance by one bin (no topology change)."""
        prev_fp = self.forward_probs[-1]

        # Compute recombination probabilities
        recomb_probs = []
        for iv in self.curr_intervals:
            dt = iv.time - self.cut_time
            recomb_probs.append(rho * dt * math.exp(-rho * dt))

        # Compute recombination weights (only full intervals contribute)
        recomb_weights = []
        for i, iv in enumerate(self.curr_intervals):
            if iv.full(self.cut_time):
                recomb_weights.append(recomb_probs[i] * iv.weight)
            else:
                recomb_weights.append(0.0)
        ws = sum(recomb_weights)
        if ws > 0:
            recomb_weights = [w / ws for w in recomb_weights]

        # Total recombination mass
        R = sum(recomb_probs[i] * prev_fp[i]
                for i in range(len(self.curr_intervals)))

        # Forward update
        new_fp = [
            prev_fp[i] * (1 - recomb_probs[i]) + R * recomb_weights[i]
            for i in range(len(self.curr_intervals))
        ]
        self.forward_probs.append(new_fp)
```

### Transfer at recombination breakpoints

When the tree topology changes, the state space itself changes — some branches disappear, others appear. The BSP must redistribute probability mass from old intervals to new ones.

Each old interval is classified by its branch:

- **Source branch** (the lineage that recombines): mass below `start_time` goes to `recombined_branch`; mass above collapses to a point mass on `merging_branch`. If the interval straddles `start_time`, split proportionally using `cc.weight()`.

- **Target branch** (where it re-coalesces): mass below `inserted_node.time` goes to `lower_transfer_branch`; mass above goes to `upper_transfer_branch`. Some mass may also go to `recombined_branch` via `_get_overwrite_prob()`.

- **Other branches**: if affected by the topology change, route to `merging_branch`; otherwise keep the same branch.

After collecting all transfers, build new `Interval` objects. Apply a **pruning cutoff**: discard partial intervals whose probability is below `self.cutoff`. This prevents the state space from growing without bound.

### Emission

After each forward/transfer step, multiply each interval's probability by its emission probability (from `PolarEmission`), then renormalise:

```python
def null_emit(self, theta, query_node):
    fp = self.forward_probs[-1]
    for i, iv in enumerate(self.curr_intervals):
        fp[i] *= self.eh.null_emit(iv.branch, iv.time, theta, query_node)
    ws = sum(fp)
    if ws <= 0:
        raise RuntimeError("BSP null_emit: forward prob sum is zero")
    for i in range(len(fp)):
        fp[i] /= ws
```

### Traceback — sampling the branch path

After the full forward pass, walk backward to sample which branch the lineage joins at each position. At each step:

1. At the rightmost position, sample an interval proportional to its forward probability.
2. Walk left: at each step, compute the probability of *not* recombining. Accumulate a running product of these "shrinkage" factors. When a uniform random draw exceeds the product, you've found a recombination event — sample a new source interval proportional to `recomb_prob * forward_prob`.
3. At transfer boundaries, sample from `source_weights` / `source_intervals`.

The result is a `SortedDict[position → Branch]` — the branch path.

> **Code:** `pysinger/hmm/bsp.py`

---

## Step 10 — The TSP (Time Sequence Propagator)

**Goal.** Conditioned on the BSP's sampled branch path, sample a coalescence **time** at each position.

### Exponential-quantile grid

For a branch $[l, u]$, the TSP discretises the time axis using the $\text{Exp}(1)$ CDF. Instead of uniformly spacing grid points, it places them at exponential quantiles:

$$
lq = 1 - e^{-l}, \quad uq = 1 - e^{-u}, \quad K = \lceil (uq - lq) / q \rceil
$$

Grid boundaries are at:

$$
t_k = -\log\!\left(1 - \left(lq + k \cdot \frac{uq - lq}{K}\right)\right), \quad k = 0, \ldots, K
$$

Why quantile spacing? The coalescent density is highest near the bottom of the branch (close to the present). Uniform spacing would waste most grid points in the sparse upper region. Quantile spacing automatically concentrates resolution where it matters — near the bottom. The gap parameter $q$ (default 0.02) controls how many grid points you get: smaller $q$ = finer grid.

```python
def _generate_grid(self, lb, ub):
    lq = 1.0 - math.exp(-lb)
    uq = 1.0 - math.exp(-ub)
    q = uq - lq
    n = math.ceil(q / self.gap)
    points = [lb]
    for i in range(1, n):
        p = lq + i * q / n
        points.append(-math.log(1.0 - p))
    points.append(ub)
    return points

def _generate_intervals(self, branch, lb, ub):
    """Create Interval objects for each adjacent pair of grid points."""
    points = self._generate_grid(lb, ub)
    for i in range(len(points) - 1):
        iv = Interval(branch, points[i], points[i + 1], self.curr_index)
        iv.fill_time()
        self.curr_intervals.append(iv)
        self._temp.append(0.0)
```

### The PSMC transition kernel

The TSP uses a **PSMC-style** transition model. Given a lineage currently at time $s$, the probability of transitioning to interval $[t_1, t_2]$ is:

$$
P_{\text{PSMC}}(\rho, s, [t_1, t_2]) = \underbrace{\mathbb{1}[t_1 \leq s \leq t_2] \cdot e^{-\rho \ell}}_{\text{no recombination: stay}} + \underbrace{\int_{t_1}^{t_2} f(\rho, s, t)\,dt}_{\text{recombine and re-coalesce}}
$$

The first term is a point mass: with probability $e^{-\rho\ell}$ (where $\ell = 2s - \text{lower\_bound} - \text{cut\_time}$), no recombination happens and the lineage stays at time $s$. The integral term accounts for recombination followed by re-coalescence at a new time.

The transition matrix is **tridiagonal**: most of the probability mass stays at the current time or moves to adjacent cells. This allows an $O(K)$ forward update instead of $O(K^2)$.

The key idea behind the $O(K)$ recursion: instead of computing all $K \times K$ transition probabilities, use cumulative sums from below (`lower_sums`) and from above (`upper_sums`):

```python
def forward(self, rho):
    """Advance one bin using tridiagonal PSMC kernel."""
    self._compute_diagonals(rho)       # stay-in-place: D[i]
    self._compute_lower_diagonals(rho) # from interval above: L[i]
    self._compute_upper_diagonals(rho) # from interval below: U[i]

    # lower_sums[i] accumulates contributions from all intervals below i
    self.lower_sums[0] = 0.0
    for i in range(1, self.dim):
        self.lower_sums[i] = (
            self.upper_diagonals[i] * self.forward_probs[-1][i - 1]
            + self.factors[i] * self.lower_sums[i - 1]
        )

    # upper_sums[i] = sum of forward probs above i
    self.upper_sums[-1] = 0.0
    for i in range(self.dim - 2, -1, -1):
        self.upper_sums[i] = (
            self.forward_probs[-1][i + 1] + self.upper_sums[i + 1]
        )

    # The full forward update in O(K)
    new_fp = []
    for i in range(self.dim):
        new_fp.append(
            self.lower_sums[i]
            + self.diagonals[i] * self.forward_probs[-1][i]
            + self.lower_diagonals[i] * self.upper_sums[i]
        )
    self.forward_probs.append(new_fp)
```

The `factors[i]` array stores ratios of exponential masses between adjacent intervals: $(e^{-lb_i} - e^{-ub_i}) / (e^{-lb_{i-1}} - e^{-ub_{i-1}})$, clamped to at most 5 to prevent numerical instability.

### Transfer at topology changes

Three cases:

1. **Source → merging**: The lineage was on the source branch; after the recombination, it's on the merging branch. Collapse all probability to a **point mass** at `deleted_node.time` — you know exactly when the coalescence happened because the deleted node is at a known time.

2. **Target → recombined**: The lineage was on the target branch; now it's on the newly created recombined branch. Expand from a point mass to the full recombined branch range, putting probability above `start_time`.

3. **Regular**: Overlap intervals by time between the old and new branches, transferring mass proportionally (in exponential measure).

### Traceback — sampling coalescence times

Walk backward, sampling an interval at each step. Convert each sampled interval to a `Node` at a jittered time:

```python
def _exp_median(self, lb, ub):
    """Sample a time uniformly in exponential quantile space."""
    if math.isinf(ub):
        return lb + 2.0 * random()
    if ub - lb <= 0.005:
        return (0.45 + 0.1 * random()) * (ub - lb) + lb
    lq = 1.0 - math.exp(-lb)
    uq = 1.0 - math.exp(-ub)
    # Jitter around the median: 0.45 to 0.55 in quantile space
    mq = (0.45 + 0.1 * random()) * (uq - lq) + lq
    return max(lb, min(ub, -math.log(1.0 - mq)))
```

The jitter (0.45 to 0.55 instead of exactly 0.5) ensures diversity across MCMC iterations — without it, you'd always pick the same time for a given interval.

> **Code:** `pysinger/hmm/tsp.py`

---

## Step 11 — The Threader

**Goal.** Combine BSP + TSP into a single threading operation.

The `Threader` orchestrates the full pipeline: run the BSP to choose branches, run the TSP to choose times, then insert the lineage into the ARG.

### Initial threading

```python
class Threader:
    def __init__(self, cutoff=0.0, gap=0.02):
        self.bsp = BSP()
        self.tsp = TSP()
        self.pe = PolarEmission()  # BSP uses polarised emission
        self.be = BinaryEmission() # TSP uses symmetric emission

    def thread(self, arg, node):
        """Add a new leaf node to the ARG."""
        arg.add_sample(node)
        self._run_bsp(arg)                     # BSP forward pass
        self._sample_joining_branches(arg)     # BSP traceback → branches
        self._run_tsp(arg)                     # TSP forward pass
        self._sample_joining_points(arg)       # TSP traceback → nodes
        arg.add(self.new_joining_branches,
                self.added_branches)           # thread into ARG
        arg.approx_sample_recombinations()     # set recombination times
        arg.clear_remove_info()
```

### The BSP forward pass loop

```python
def _run_bsp(self, arg):
    self.bsp.set_emission(self.pe)  # PolarEmission for branch selection
    self.bsp.start(arg.start_tree, self.cut_time)

    for i in range(self.start_index, self.end_index):
        pos = arg.coordinates[i]
        query_node = arg.get_query_node_at(pos)

        # Transition step: topology change or regular forward
        if pos is at a recombination breakpoint:
            self.bsp.transfer(recombination)
        elif pos != self.start:
            self.bsp.forward(arg.rhos[i - 1])

        # Collect mutations in [pos, next_pos)
        next_pos = arg.coordinates[i + 1]
        mut_set = {m for m in arg.mutation_sites
                   if pos <= m < next_pos}

        # Emission step
        if mut_set:
            self.bsp.mut_emit(arg.thetas[i], next_pos - pos,
                             mut_set, query_node)
        else:
            self.bsp.null_emit(arg.thetas[i], query_node)
```

### The TSP forward pass loop

Same structure, but conditioned on the BSP's sampled joining branches. The TSP uses `BinaryEmission` and handles three transition types: `transfer` (at recombination), `recombine` (branch change without topology change), and `forward` (same branch).

### MCMC re-threading with Metropolis acceptance

```python
def internal_rethread(self, arg, cut_point):
    """MCMC move: remove a lineage and propose a new threading."""
    self.cut_time = cut_point[2]
    arg.remove(cut_point)          # extract lineage
    self._run_bsp(arg)             # propose new branches
    self._sample_joining_branches(arg)
    self._run_tsp(arg)             # propose new times
    self._sample_joining_points(arg)

    # Metropolis acceptance ratio
    ar = self._acceptance_ratio(arg)
    if random() < ar:
        arg.add(self.new_joining_branches,
                self.added_branches)    # ACCEPT proposal
    else:
        arg.add(arg.joining_branches,
                arg.removed_branches)   # REJECT: restore original

    arg.approx_sample_recombinations()
    arg.clear_remove_info()
```

### The acceptance ratio

$$
\alpha = \frac{h_{\text{old}}}{h_{\text{new}}}
$$

where $h$ is the **effective tree height** at the cut position — the maximum child-node time in the cut tree. If the joining branch reaches the root sentinel, use the coalescence node's time instead.

This ratio favours proposals that produce comparable or shorter trees. Proposals that inflate the tree height get penalised (ratio < 1, lower acceptance probability). This keeps the MCMC from drifting toward unreasonably tall trees.

```python
def _acceptance_ratio(self, arg):
    cut_height = max(child.time
                     for child in arg.cut_tree.parents.keys())
    old_height = cut_height
    new_height = cut_height

    # Adjust if joining branch reaches root
    if old_joining_branch.upper_node is arg.root:
        old_height = old_added_branch.upper_node.time
    if new_joining_branch.upper_node is arg.root:
        new_height = new_added_branch.upper_node.time

    if new_height <= 0:
        return 1.0
    return old_height / new_height
```

> **Code:** `pysinger/mcmc/threader.py`

---

## Step 12 — The Sampler

**Goal.** Top-level MCMC orchestrator.

```python
class Sampler:
    def __init__(self, Ne=1.0, recomb_rate=0.0, mut_rate=0.0):
        self.Ne = Ne
        self.recomb_rate = recomb_rate * Ne  # scale to coalescent units
        self.mut_rate = mut_rate * Ne
        self.bsp_c = 0.0   # BSP pruning cutoff
        self.tsp_q = 0.02  # TSP grid gap
```

### Loading data

```python
def load_vcf(self, vcf_file, start=0, end=float("inf"), haploid=False):
    if haploid:
        nodes, seq_len = read_vcf_haploid(vcf_file, start, end)
    else:
        nodes, seq_len = read_vcf_phased(vcf_file, start, end)
    self.sequence_length = seq_len
    self.sample_nodes = set(nodes)
    # Shuffle threading order for better initial ARGs
    self.ordered_sample_nodes = shuffled(nodes, seed=self._seed)
```

### Building the initial ARG

```python
def iterative_start(self, max_retries=5):
    """Thread all samples one-by-one to build an initial ARG."""
    for attempt in range(max_retries):
        try:
            self._build_singleton_arg()        # ARG with first sample
            for node in self.ordered_sample_nodes[1:]:
                threader = self._make_threader()
                threader.thread(self.arg, node) # thread each sample
            self._rescale()                    # calibrate times
            return
        except RuntimeError:
            # HMM underflow: retry with different RNG
            self._rng = np.random.default_rng(self._seed + attempt + 1)
```

### The MCMC loop

```python
def internal_sample(self, num_iters, spacing=1):
    """Run MCMC iterations. Each iteration re-threads enough lineages
    to cover at least spacing * sequence_length base pairs."""
    for iteration in range(num_iters):
        updated_length = 0.0
        while updated_length < spacing * self.arg.sequence_length:
            threader = self._make_threader()
            cut_point = self.arg.sample_internal_cut()
            try:
                threader.internal_rethread(self.arg, cut_point)
            except Exception:
                # Restore ARG on failure
                if self.arg.joining_branches:
                    self.arg.add(self.arg.joining_branches,
                                self.arg.removed_branches)
                    self.arg.approx_sample_recombinations()
                self.arg.clear_remove_info()
                break
            updated_length += (threader.end - threader.start)
            self.arg.clear_remove_info()
        self._rescale()
```

### Rescaling — mutation-rate calibration

After each MCMC iteration, all coalescence times are rescaled so the expected number of mutations matches the observed count. The idea: if the ARG's total branch length predicts more mutations than we observe, the tree is too tall — scale it down. If it predicts fewer, scale it up.

$$
s = \frac{S_{\text{obs}}}{\mu_{\text{scaled}} \cdot L_{\text{total}}}
$$

where:
- $S_{\text{obs}}$ = number of unique segregating site positions across all samples
- $\mu_{\text{scaled}} = \mu \cdot N_e$ = mutation rate in coalescent units
- $L_{\text{total}}$ = total ARG branch length from `get_arg_length()`

```python
def _rescale(self):
    total_obs = len({pos for n in self.sample_nodes
                     for pos in n.mutation_sites if pos >= 0})
    total_branch = self.arg.get_arg_length()
    expected = self.mut_rate * total_branch
    if expected <= 0:
        return 1.0
    scale = total_obs / expected

    # Discover internal nodes by walking all trees
    internal_nodes = []
    tree = self.arg.get_tree_at(0.0)
    sample_ids = {id(n) for n in self.sample_nodes}
    seen = set()
    for n in tree.parents.values():
        if n.index >= 0 and id(n) not in seen and id(n) not in sample_ids:
            seen.add(id(n))
            internal_nodes.append(n)
    for pos, r in self.arg.recombinations.items():
        if 0 < pos < self.arg.sequence_length:
            tree.forward_update(r)
            for n in tree.parents.values():
                if n.index >= 0 and id(n) not in seen and id(n) not in sample_ids:
                    seen.add(id(n))
                    internal_nodes.append(n)

    # Rescale all internal node times
    for n in internal_nodes:
        n.time *= scale

    # Also rescale recombination start_times
    for pos, r in self.arg.recombinations.items():
        if 0 < pos < self.arg.sequence_length and r.start_time > 0:
            r.start_time *= scale

    return scale
```

> **Code:** `pysinger/sampler.py`

---

## Step 13 — VCF reader

**Goal.** Parse phased genotypes into `Node` objects.

```python
def read_vcf_phased(vcf_file, start_pos=0.0, end_pos=float("inf")):
    """Read a phased VCF and return (nodes, sequence_length).

    Creates 2 Node objects per diploid individual (one per haplotype).
    """
    nodes = []

    with open(vcf_file) as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                # Count individuals from header
                fields = line.split()
                n_ind = len(fields) - 9
                nodes = [Node(time=0.0, index=i)
                         for i in range(2 * n_ind)]
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            pos = float(parts[1])
            ref, alt = parts[3], parts[4]

            # Skip if outside region, duplicate, or indel
            if pos < start_pos or pos > end_pos:
                continue
            if len(ref) > 1 or len(alt) > 1:
                continue

            # Parse genotypes: "0|1" -> alleles [0, 1]
            genotypes = parts[9:]
            gt_vals = []
            for g in genotypes:
                gt_vals.append(1 if g[0] == "1" else 0)
                gt_vals.append(1 if (len(g) > 2 and g[2] == "1") else 0)

            # Only keep segregating sites (not fixed ancestral or derived)
            gt_sum = sum(gt_vals)
            if 1 <= gt_sum < len(gt_vals):
                rel_pos = pos - start_pos
                for i, v in enumerate(gt_vals):
                    if v == 1:
                        nodes[i].add_mutation(rel_pos)

    sequence_length = end_pos - start_pos
    return nodes, sequence_length
```

Key points:
- Positions are stored **relative to `start_pos`** so the ARG always starts at coordinate 0.
- **Fixed sites** (all ancestral or all derived) are skipped — they carry no information about the genealogy.
- **Indels and multi-allelic sites** are skipped — SINGER assumes biallelic SNPs.

> **Code:** `pysinger/io/vcf_reader.py`

---

## Step 14 — Fitch parsimony reconstruction

**Goal.** Assign ancestral states to internal nodes with the minimum number of mutations.

When the ARG threads a new lineage, the newly created coalescence nodes need allele states. Fitch parsimony does this optimally (minimum mutations) in two passes:

### Pass 1 — Pruning (bottom-up)

Start at the leaves and work up. Each leaf reads its state from `node.get_state(pos)`. For each internal node, merge its two children:

```python
def _fitch_up(self, c1, c2, parent):
    s1 = self.pruning_states[c1]
    s2 = self.pruning_states[c2]
    if s1 == s2:
        self.pruning_states[parent] = s1      # children agree
    elif s1 == 0.5:
        self.pruning_states[parent] = s2      # c1 is ambiguous
    elif s2 == 0.5:
        self.pruning_states[parent] = s1      # c2 is ambiguous
    else:
        self.pruning_states[parent] = 0.5     # disagree: ambiguous
```

The value 0.5 means "ambiguous — could be either 0 or 1." When children disagree, a mutation must have occurred on one of the branches, but we don't know which — so we defer the decision to the top-down pass.

### Pass 2 — Peeling (top-down)

Start at the root and work down, resolving ambiguities using the parent's definite state:

```python
def _fitch_down(self, parent, child):
    if parent.index == -1:
        # Root sentinel: resolve ambiguity to ancestral (0)
        s = 0.0 if self.pruning_states[child] == 0.5 \
            else self.pruning_states[child]
        self.peeling_states[child] = s
        return

    sp = self.peeling_states[parent]  # parent's resolved state
    sc = self.pruning_states[child]   # child's pruning state
    if sc == 0.5:
        self.peeling_states[child] = sp  # inherit parent's state
    else:
        self.peeling_states[child] = sc  # keep definite state
```

After both passes, write the resolved states back: `node.write_state(pos, state)` for every internal node.

```python
def reconstruct(self, pos):
    self.pruning_states = {}
    self.peeling_states = {}
    for n in self.node_set:
        self._pruning_pass(n)   # bottom-up
    for n in self.node_set:
        self._peeling_pass(n)   # top-down
    for n, s in self.peeling_states.items():
        n.write_state(pos, s)   # write back
```

> **Code:** `pysinger/reconstruction/fitch.py`

---

## Step 15 — tskit export

**Goal.** Convert the final ARG to a `tskit.TreeSequence` for downstream analysis.

```python
import tskit

def arg_to_tskit(arg, Ne=1.0):
    tables = tskit.TableCollection(sequence_length=arg.sequence_length)

    # 1. Discover all nodes by replaying the full ARG
    node_map = {}  # id(pysinger_node) -> tskit_node_id
    tree = arg.get_tree_at(0.0)

    def collect(tree_obj):
        for child, parent in tree_obj.parents.items():
            for n in (child, parent):
                if n.index != -1 and id(n) not in node_map:
                    is_sample = n in arg.sample_nodes
                    tskit_id = tables.nodes.add_row(
                        flags=tskit.NODE_IS_SAMPLE if is_sample else 0,
                        time=n.time * Ne,
                    )
                    node_map[id(n)] = tskit_id

    collect(tree)
    for pos, r in arg.recombinations.items():
        if 0 < pos < arg.sequence_length:
            tree.forward_update(r)
            collect(tree)

    # 2. Emit edges for each tree interval
    tree = arg.get_tree_at(0.0)
    prev_pos = 0.0
    recomb_positions = sorted(
        p for p in arg.recombinations if 0 < p < arg.sequence_length
    )

    def emit_edges(tree_obj, left, right):
        for child, parent in tree_obj.parents.items():
            if parent.index == -1 or parent.time <= child.time:
                continue
            c_id = node_map.get(id(child))
            p_id = node_map.get(id(parent))
            if c_id is not None and p_id is not None:
                tables.edges.add_row(
                    left=left, right=right, parent=p_id, child=c_id
                )

    for rpos in recomb_positions:
        emit_edges(tree, prev_pos, rpos)
        tree.forward_update(arg.recombinations[rpos])
        prev_pos = rpos
    emit_edges(tree, prev_pos, arg.sequence_length)

    # 3. Sort and build
    tables.sort()
    return tables.tree_sequence()
```

The time conversion `n.time * Ne` converts from coalescent units (where 1 unit = $N_e$ generations) back to generations, which is what tskit expects. The resulting tree sequence can be used with the full tskit API for computing diversity, TMRCA, drawing trees, etc.

> **Code:** `pysinger/io/tskit_writer.py`

---

## Putting it all together

```python
from pysinger import Sampler
from pysinger.io.tskit_writer import arg_to_tskit

sampler = Sampler(Ne=10000, recomb_rate=1e-8, mut_rate=1e-8)
sampler.load_vcf("data.vcf", start=0, end=1_000_000)
sampler.iterative_start()
sampler.internal_sample(num_iters=100, spacing=1)
ts = arg_to_tskit(sampler.arg, Ne=10000)
```

### Dependency graph

```
Steps 1-2 (Node, Branch)
    └── Step 3 (Tree)
    └── Step 4 (Recombination)
    └── Step 5 (Interval)
        └── Step 6 (ARG)
            └── Step 7 (CoalescentCalculator)
            └── Step 8 (Emission models)
                └── Step 9 (BSP)
                └── Step 10 (TSP)
                    └── Step 11 (Threader)
                        └── Step 12 (Sampler)
Step 13 (VCF reader)  ← independent, needed by Sampler
Step 14 (Fitch)       ← used by ARG._impute()
Step 15 (tskit writer) ← used after sampling
```
