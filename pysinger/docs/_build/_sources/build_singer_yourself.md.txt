# Build SINGER yourself

This chapter walks you through implementing SINGER from scratch. Each step tells you exactly which class or function to write next, which equations drive it, and how the pieces connect. By the end you will have a working Bayesian ARG sampler.

The guide follows the dependency order: data structures first, then the math helpers, then the two HMMs, then the MCMC loop, and finally I/O. Each section ends with a short pointer to the corresponding file in the pysinger codebase.

---

## Step 1 — The Node class

**Goal.** Represent a single vertex in the ARG.

A `Node` stores three things:

1. **`time: float`** — the coalescence time in coalescent units. Leaf nodes (observed haplotypes) have `time = 0`. Internal nodes have `time > 0`. You will also need a **root sentinel** with `time = inf` and `index = -1`; every tree in the ARG hangs off this sentinel.

2. **`index: int`** — an integer label. Samples get indices `0, 1, …, n-1`; internal nodes get larger indices; the root sentinel gets `-1`.

3. **`mutation_sites: SortedDict[float → int]`** — a sorted map from genomic position to allele state (`1` = derived, `0` = ancestral). Insert a **sentinel entry at position `-1` with state `0`** so that any lookup before the first real site returns the ancestral state without special-casing.

Write three methods:

- **`add_mutation(pos)`** — set `mutation_sites[pos] = 1`.
- **`get_state(pos)`** — return the stored state at `pos` if the key exists exactly, otherwise `0`. Mutations are point events: a mutation at position 100 does *not* carry its state forward to position 101. This is a simple `.get(pos, 0)` call on the sorted dict.
- **`write_state(pos, state)`** — overwrite the state; if `state == 0`, remove the entry (but never remove the sentinel at `-1`).

**Critical design decision: identity semantics.** Override `__eq__` to use `self is other` and `__hash__` to return `id(self)`. Two `Node` objects with identical fields are still different nodes. This mirrors C++ pointer comparison and is essential because the same coalescence time can appear in multiple nodes — you must distinguish them by identity, not by value.

> **Code:** `pysinger/data/node.py`

---

## Step 2 — The Branch class

**Goal.** Represent a directed edge between two nodes.

A `Branch` holds two `Node` references:

- **`lower_node`** — the child (lower time)
- **`upper_node`** — the parent (higher time)

with the invariant `lower_node.time ≤ upper_node.time`.

Write the class with these properties:

1. **Immutability.** Override `__setattr__` to raise `AttributeError` after construction. Use `object.__setattr__` inside `__init__` to set the two fields. Branches are used as dict keys and set members, so they must not change after creation.

2. **Identity-based hashing.** `__hash__` returns `hash((id(self.lower_node), id(self.upper_node)))`. `__eq__` checks `self.lower_node is other.lower_node and self.upper_node is other.upper_node`. This matches the C++ `branch_hash` struct.

3. **Null branch.** `Branch()` with both nodes as `None` serves as a sentinel (e.g., to mark the end of a removed lineage). `__bool__` returns `False` for null branches.

4. **`length` property.** Returns `upper_node.time - lower_node.time`, or `inf` if either node is `None`.

> **Code:** `pysinger/data/branch.py`

---

## Step 3 — The Tree class

**Goal.** Represent a marginal coalescent tree at a single genomic position.

The tree is stored as two dictionaries:

```python
parents:  Dict[Node, Node]       # child → parent
children: Dict[Node, Set[Node]]  # parent → {child₁, child₂}
```

Write these methods:

### `insert_branch(branch)` and `delete_branch(branch)`

These add or remove a single branch from the two dicts. `insert_branch` sets `parents[lower] = upper` and adds `lower` to `children[upper]`. `delete_branch` reverses this. These are the atomic operations that all topology changes are built from.

### `forward_update(recombination)` and `backward_update(recombination)`

Apply a `Recombination` record to move the tree forward (or backward) along the genome:

```python
def forward_update(self, r):
    for b in r.deleted_branches:
        self.delete_branch(b)
    for b in r.inserted_branches:
        self.insert_branch(b)
```

`backward_update` does the reverse (delete inserted, re-insert deleted).

### `find_sibling(node)` and `find_joining_branch(removed_branch)`

These are key for the MCMC. Given a node `n`, `find_sibling` returns the other child of `n`'s parent. Given a branch being removed, `find_joining_branch` returns `Branch(sibling, grandparent)` — the branch that "takes over" when the removed branch's coalescence node is pruned from the tree. This is the SPR operation in reverse.

### `length()`

Sum of all branch lengths (excluding branches to the root sentinel at `index == -1`). Used later by the rescaling step.

> **Code:** `pysinger/data/tree.py`

---

## Step 4 — The Recombination class

**Goal.** Record a single topology change at a genomic breakpoint.

A `Recombination` at position `pos` stores:

- **`deleted_branches`**: set of branches that exist *before* (left of) `pos`
- **`inserted_branches`**: set of branches that exist *after* (right of) `pos`

From these two sets, the class derives several named branches that the HMMs need. Write a `_find_nodes()` method that identifies:

- **`deleted_node`**: the upper node that appears in `deleted_branches` but not in `inserted_branches` (the old coalescence node being removed)
- **`inserted_node`**: the upper node that appears in `inserted_branches` but not in `deleted_branches` (the new coalescence node being created)

Then write `_find_target_branch()` to identify which deleted branch was split by `inserted_node`:

```
For each deleted branch b (other than source_branch):
    if b.lower_node.time ≤ inserted_node.time ≤ b.upper_node.time:
        lower = Branch(b.lower_node, inserted_node)
        upper = Branch(inserted_node, b.upper_node)
        if lower ∈ inserted_branches AND upper ∈ inserted_branches:
            target_branch = b
```

Then write `_find_recomb_info()` to compute:

| Derived field | Definition |
|---|---|
| `merging_branch` | `Branch(sibling_lower, grandparent_upper)` — the branch formed after `deleted_node` is removed |
| `recombined_branch` | `Branch(source_branch.lower_node, inserted_node)` — the source lineage below the recombination height, re-attached to `inserted_node` |
| `lower_transfer_branch` | `Branch(target_branch.lower_node, inserted_node)` — target below the new node |
| `upper_transfer_branch` | `Branch(inserted_node, target_branch.upper_node)` — target above the new node |
| `start_time` | Height at which the recombination occurs (set later by `approx_sample_recombinations`) |

### `trace_forward(t, branch)` and `trace_backward(t, branch)`

These map a branch at time `t` across the topology change. The BSP and MCMC use them to follow a lineage through recombination events:

```
trace_forward(t, branch):
    if branch is not affected by this recombination: return branch
    if branch == source_branch:
        if t ≥ start_time: return null  (lineage was cut above)
        else: return recombined_branch
    if branch == target_branch:
        if t > inserted_node.time: return upper_transfer_branch
        else: return lower_transfer_branch
    otherwise: return merging_branch
```

### `_simplify_branches()`

After any modification, remove branches that appear in *both* `deleted_branches` and `inserted_branches` — they cancel out.

### `remove()` and `add()`

These complex methods update a `Recombination` record when a lineage is being removed from or threaded into the ARG. They add/remove pairs of branches from both sets, then re-simplify and re-derive the named fields. The logic swaps old branches (around the removed/added node) for new ones (after the topology surgery). Study the implementation carefully — the branch swapping follows the SPR geometry.

> **Code:** `pysinger/data/recombination.py`

---

## Step 5 — The Interval and IntervalInfo classes

**Goal.** Represent a single cell `(branch, [lb, ub])` in the HMM state space.

An `Interval` stores:

- **`branch`**: which branch in the tree
- **`lb, ub`**: lower and upper time bounds of the cell
- **`start_pos`**: the HMM step index where this interval was created (used during traceback to know when to switch to transfer-sourced intervals)
- **`weight`**: coalescent probability mass $F(ub) - F(lb)$ (set by `CoalescentCalculator`)
- **`time`**: representative time point within the cell (set by `fill_time()`)
- **`source_weights` / `source_intervals`**: lists storing traceback pointers from BSP transfer steps
- **`node`**: optional `Node` pointer (used by TSP for point-mass intervals at known coalescence nodes)

### `fill_time()`

Compute the **exponential median** of `[lb, ub]`. This is the time $t$ where the Exp(1) CDF is midway between $F(lb)$ and $F(ub)$:

$$
lq = 1 - e^{-lb}, \quad uq = 1 - e^{-ub}, \quad q = \frac{lq + uq}{2}, \quad t = -\log(1 - q)
$$

Handle edge cases: if `ub` is infinite, return `lb + log(2)`. If the interval is tiny (`|ub - lb| < 1e-3`), return the arithmetic midpoint.

### `full(cut_time)`

Returns `True` if this interval spans the entire branch above `cut_time`. This matters because only "full" intervals participate in the BSP recombination weight calculation.

**`IntervalInfo`** is a lightweight hashable key `(branch, lb, ub, seed_pos)` used as dict keys inside the BSP transfer step to accumulate probability mass from multiple source intervals into the same target region.

> **Code:** `pysinger/data/interval.py`

---

## Step 6 — The ARG class

**Goal.** The central data structure encoding the entire Ancestral Recombination Graph.

An ARG is a **sorted map** `SortedDict[position → Recombination]` with sentinel entries at position `0` and `INT_MAX`. The marginal tree at any position $x$ is obtained by replaying all recombination records from `0` up to $x$:

```python
def get_tree_at(self, x):
    tree = Tree()
    for pos, r in self.recombinations.items():
        if pos <= x:
            tree.forward_update(r)
        else:
            break
    return tree
```

### Constructor

Store: `Ne`, `sequence_length`, a root sentinel node (`Node(time=inf, index=-1)`), empty `sample_nodes` and `node_set` sets, the recombination map with two sentinels, and empty mutation tracking structures (`mutation_sites`, `mutation_branches`).

### `build_singleton_arg(node)`

Create the simplest possible ARG: one sample `node` connected to the root sentinel by a single branch. The recombination at position `0` gets `inserted_branches = {Branch(node, root)}`.

### `discretize(bin_size)` — the HMM grid

Build a coordinate grid along the genome. Start at position 0, advance by `bin_size`, but always place a grid point exactly at each recombination breakpoint:

```python
while curr_pos < sequence_length:
    coordinates.append(curr_pos)
    next_recomb = ...  # next recombination position
    if next_recomb < curr_pos + bin_size:
        curr_pos = next_recomb  # snap to breakpoint
    else:
        curr_pos += bin_size
coordinates.append(sequence_length)
```

### `compute_rhos_thetas(r, m)`

For each bin `[coordinates[i], coordinates[i+1])`, compute:
- `rhos[i] = r × span` (scaled recombination rate)
- `thetas[i] = m × span` (scaled mutation rate)

where `r` and `m` are already in coalescent units (multiplied by $N_e$).

### `remove(cut_point)` — MCMC lineage extraction

This is the most complex method. Given `(pos, center_branch, cut_time)`:

1. Create a `cut_node` sentinel at `cut_time` (with `index = -2`).
2. **Forward pass**: starting from `pos`, trace `center_branch` rightward through successive recombination records using `trace_forward(cut_time, branch)`. At each record, call `r.remove(...)` to update the topology, and store the removed and joining branches.
3. **Backward pass**: starting from `pos`, trace leftward using `trace_backward(cut_time, branch)`. Same logic in reverse.
4. Clean up empty recombination records, remap mutations, and update the cut tree.

### `add(joining_branches, added_branches)` — MCMC lineage insertion

Walk through the `added_branches` map. At each position, either update an existing recombination record via `r.add(...)` or create a new one. Then call `_impute()` to assign allele states to the newly created coalescence nodes.

### `_impute(joining_branches, added_branches)`

For each genomic interval, look up which mutations fall in it, then call `_map_mutation_branch()` for each one. This uses majority-rule: given the states of the lower node ($s_l$), upper node ($s_u$), and query node ($s_0$), the new coalescence node gets state $s_m = \mathbb{1}[s_l + s_u + s_0 > 1]$.

### `get_arg_length()`

Replay all trees, summing `tree.length() × genomic_span` for each interval. Used by the rescaling step.

### `sample_internal_cut()`

Sample a random `(pos, branch, time)` for the next MCMC move. Draw `time` uniformly over the tree height, then find the branch spanning that time.

### `approx_sample_recombinations()`

After threading, assign `start_time` to every recombination record. For each record, find the source branch (the deleted branch whose lower node connects to `inserted_node` via `recombined_branch`), then sample `start_time` uniformly in the valid range `[max(cut_time, source.lower_node.time), min(source.upper_node.time, inserted_node.time)]`.

> **Code:** `pysinger/data/arg.py`

---

## Step 7 — The CoalescentCalculator

**Goal.** Compute the piecewise-exponential coalescent CDF for a set of branches.

Given $m$ branches, the coalescence rate at time $t$ is:

$$
\lambda(t) = \#\{\text{branches alive at time } t\}
$$

The CDF is:

$$
F(t) = 1 - \exp\!\left(-\int_0^t \lambda(s)\,ds\right)
$$

Since $\lambda$ is piecewise-constant (changing only when branches start or end), $F$ is piecewise-exponential.

### `compute(branches)` — build the CDF

Do this in three sub-steps:

**Step 7a: `_compute_rate_changes(branches)`**

For each branch `b`, record `+1` at `max(cut_time, b.lower_node.time)` and `-1` at `b.upper_node.time` in a sorted dict `rate_changes`. This gives you the rate-change times.

**Step 7b: `_compute_rates()`**

Running sum of `rate_changes` → piecewise-constant rate function stored in sorted dict `rates[time] = current_rate`.

**Step 7c: `_compute_probs_quantiles()`**

Walk adjacent pairs of rate-change times. Between times $[t_k, t_{k+1})$ with rate $\lambda_k$:

$$
S(t_{k+1}) = S(t_k) \cdot e^{-\lambda_k(t_{k+1} - t_k)}
$$

Accumulate $F(t_{k+1}) = F(t_k) + S(t_k) - S(t_{k+1})$. Store in sorted dict `cum_probs[time] = F(time)`. Also build parallel arrays `(prob_vals, prob_times)` sorted by probability value for the quantile function.

### `prob(x)` — interpolated CDF lookup

Find the interval $[t_k, t_{k+1})$ containing $x$, then interpolate:

$$
F(x) = F(t_k) + \Delta F \cdot \frac{\text{expm1}(-\lambda_k \cdot (x - t_k))}{\text{expm1}(-\lambda_k \cdot (t_{k+1} - t_k))}
$$

This is exact (not an approximation) because the rate is constant within the interval.

### `quantile(p)` — inverse CDF

Given probability $p$, find the interval where $F$ crosses $p$, then invert:

$$
t = t_k - \frac{1}{\lambda_k} \log\!\left(1 - \frac{p - F(t_k)}{F(t_{k+1}) - F(t_k)} \cdot (1 - e^{-\lambda_k \Delta t})\right)
$$

### `weight(lb, ub)` and `time(lb, ub)`

- `weight(lb, ub) = prob(ub) - prob(lb)` — probability of coalescence in $[lb, ub]$.
- `time(lb, ub)` — the **exponential median**: find $t$ such that $F(t) = \frac{1}{2}[F(lb) + F(ub)]$ using `quantile(midpoint_prob)`. Falls back to the arithmetic midpoint for tiny intervals.

> **Code:** `pysinger/hmm/coalescent.py`

---

## Step 8 — Emission models

**Goal.** Compute the likelihood ratio of the data with vs. without the threaded lineage.

When you thread a new lineage onto branch $b$ at time $t$, the original branch $(l, u)$ is split into three sub-branches:

$$
\ell_{\text{lower}} = t - l, \quad \ell_{\text{upper}} = u - t, \quad \ell_{\text{query}} = t - t_q
$$

where $t_q$ is the query node's time (= 0 for leaf nodes during initial threading, or `cut_time` during MCMC).

### `BinaryEmission` (used by the TSP)

Write an abstract base class `Emission` with two methods: `null_emit(branch, time, theta, node)` and `mut_emit(branch, time, theta, bin_size, mut_set, node)`.

**Null emission** (no mutations in the bin):

$$
e_{\text{null}} = \frac{e^{-\theta \ell_l} \cdot e^{-\theta \ell_u} \cdot e^{-\theta \ell_q}}{e^{-\theta(\ell_l + \ell_u)}} = e^{-\theta \ell_q}
$$

The ratio cancels the shared Poisson no-mutation terms, leaving only the query branch's contribution.

**Mutation emission**: For each segregating site in the bin, compute the **majority-rule ancestral state**:

$$
s_m = \mathbb{1}[s_l + s_u + s_0 > 1.5]
$$

where $s_l$ = state of the lower node, $s_u$ = state of the upper node, $s_0$ = state of the query node. Then count state changes on each sub-branch: $d_l = |s_m - s_l|$, $d_u = |s_m - s_u|$, $d_q = |s_m - s_0|$, and for the original unsplit branch $d_{\text{old}} = |s_l - s_u|$.

The emission probability is a product of Poisson terms:

$$
e_{\text{mut}} = \prod_{\text{sites}} \frac{e^{-\theta \ell_l} (\theta/\Delta x)^{d_l} \cdot e^{-\theta \ell_u} (\theta/\Delta x)^{d_u} \cdot e^{-\theta \ell_q} (\theta/\Delta x)^{d_q}}{e^{-\theta(\ell_l+\ell_u)} (\theta/\Delta x)^{d_{\text{old}}}}
$$

Implement this as a helper `_calculate_prob(theta, bin_size, s)` that returns $e^{-\theta} \cdot (\theta/\text{bin\_size})^s$, and a `_get_diff(mut_set, branch, node)` that returns `[d_l, d_u, d_q, d_old]`.

### `PolarEmission` (used by the BSP)

Extends `BinaryEmission` with two refinements:

1. **Derived allele penalty**: if $d_q \geq 1$ (the query node differs from the majority), multiply by a penalty factor (default 1.0).
2. **Root reward**: when the branch reaches the root sentinel and the majority state is derived ($s_m = 0$) while the lower node is derived ($s_l = 1$), apply a factor of `ancestral_prob / (1 - ancestral_prob)` to favour the ancestral state at the root.

The null emission simplifies: for a finite branch, return $e^{-\theta \ell_q}$. For a root branch (infinite upper), return $e^{-\theta(\ell_l + \ell_q)}$.

> **Code:** `pysinger/hmm/emission.py`

---

## Step 9 — The BSP (Branch Sequence Propagator)

**Goal.** A forward HMM that answers: **which branch** should the new lineage join at each genomic position?

### State space

The state space at each position is a list of `Interval` objects, one per (branch, time-range) cell in the current marginal tree. Each interval `i` has a forward probability $\alpha_i(x)$.

### `start(branches, cut_time)` — initialisation

1. Filter to branches with `upper_node.time > cut_time`.
2. Build a `CoalescentCalculator` from these branches.
3. For each valid branch, create an `Interval(branch, lb, ub)` where `lb = max(branch.lower_node.time, cut_time)` and `ub = branch.upper_node.time`.
4. Set initial forward probability to the coalescent weight: $\alpha_i(0) = F(ub_i) - F(lb_i)$.

### `forward(rho)` — no topology change

This implements the BSP transition equation:

$$
\alpha_i(x+1) = \alpha_i(x) \cdot (1 - p_{\text{recomb},i}) + R \cdot w_i
$$

Write three helper methods:

**`_compute_recomb_probs(rho)`**: For each interval $i$, compute the recombination probability:

$$
p_{\text{recomb},i} = \rho \cdot (t_i - t_c) \cdot e^{-\rho(t_i - t_c)}
$$

where $t_i$ is the interval's representative time and $t_c$ is the cut time. This is the probability that a recombination event at rate $\rho$ hits a branch of length $t_i - t_c$.

**`_compute_recomb_weights(rho)`**: For each "full" interval (one that spans the entire branch above `cut_time`), set the recombination weight to `recomb_prob × coalescent_weight`. Normalise so they sum to 1. Non-full intervals (partial intervals created by transfer) get weight 0 — they can receive mass via the $R \cdot w_i$ term but don't contribute to $R$.

**The forward update itself**: Compute $R = \sum_j p_{\text{recomb},j} \cdot \alpha_j(x)$ (total recombination mass), then for each interval $i$:

```python
new_fp[i] = prev_fp[i] * (1 - recomb_probs[i]) + R * recomb_weights[i]
```

### `transfer(recombination)` — topology change

When the tree topology changes at a recombination breakpoint, the state space itself changes. The BSP must redistribute probability mass from old intervals to new ones.

Write `_process_interval(r, i, transfer_weights, transfer_intervals)` that classifies each old interval by its branch:

**Source branch** (the lineage that recombines): mass below `start_time` → `recombined_branch`; mass above → collapse to a point mass on `merging_branch` at `source_branch.upper_node.time`. If the interval straddles `start_time`, split the mass proportionally using `cc.weight(lb, start_time)` and `cc.weight(start_time, ub)`.

**Target branch** (where it re-coalesces): mass below `inserted_node.time` → `lower_transfer_branch`; mass above → `upper_transfer_branch`. If the interval straddles the join point, also route some mass to `recombined_branch` via `_get_overwrite_prob()` (the probability that the lineage actually came from this recombination event rather than being an unrelated lineage on the same branch).

**Other branches**: if the branch is affected (in `deleted_branches`), route to `merging_branch` with the same time bounds. Otherwise keep the same branch.

**`_generate_intervals()`**: collect all the transfer targets, create new `Interval` objects, and set their forward probabilities. Apply a **pruning cutoff**: discard partial intervals whose accumulated probability is below `self.cutoff`. Full intervals (spanning the entire branch) are always kept. This prevents the state space from growing unboundedly.

### `null_emit(theta, query_node)` and `mut_emit(theta, bin_size, mut_set, query_node)`

After each forward/transfer step, multiply each interval's forward probability by the emission probability from the `PolarEmission` model, then renormalise so probabilities sum to 1.

### `sample_joining_branches(start_index, coordinates)` — traceback

Walk backward through the forward probabilities. At the rightmost position, sample an interval proportional to its forward probability. Then at each earlier step:

1. **`_trace_back_helper(interval, x)`**: random walk backward. At each step, compute the probability of *not* recombining as `non_recomb_prob / (non_recomb_prob + recomb_contribution)`. Multiply a running product by this shrinkage factor; stop when a uniform random draw exceeds it.

2. If you stopped at a transfer boundary (`x == interval.start_pos`), sample a source interval from `interval.source_weights` / `interval.source_intervals`.

3. If you stopped due to recombination, sample a new source interval proportional to `recomb_prob × forward_prob`.

The result is a `SortedDict[position → Branch]`. Simplify consecutive identical branches.

> **Code:** `pysinger/hmm/bsp.py`

---

## Step 10 — The TSP (Time Sequence Propagator)

**Goal.** Conditioned on the BSP's sampled branch path, sample a coalescence time at each position.

### State space — exponential-quantile grid

For a branch $[l, u]$, discretise the time axis using the $\text{Exp}(1)$ CDF:

$$
lq = 1 - e^{-l}, \quad uq = 1 - e^{-u}, \quad K = \lceil (uq - lq) / q \rceil
$$

where $q$ is the gap parameter (default 0.02). Place grid boundaries at:

$$
t_k = -\log\!\left(1 - \left(lq + k \cdot \frac{uq - lq}{K}\right)\right), \quad k = 0, \ldots, K
$$

This spacing puts more resolution where the coalescent density is highest (near the bottom of the branch) and fewer points near the top. Each adjacent pair of boundaries defines one `Interval`.

Write `_generate_grid(lb, ub)` that returns this list of boundary points, and `_generate_intervals(branch, lb, ub)` that creates `Interval` objects for each adjacent pair.

### `start(branch, cut_time)` — initialisation

Generate the grid on the starting branch, set initial forward probabilities to $e^{-lb_i} - e^{-ub_i}$ (unnormalised Exp(1) probabilities for each cell).

### `forward(rho)` — PSMC transition kernel

The TSP uses a **PSMC-style** tridiagonal transition matrix. For each interval $i$ with representative time $t_i$, compute:

**Diagonal** (stay in same interval):

$$
D_i = \frac{P_{\text{PSMC}}(\rho, t_i, [lb_i, ub_i])}{P_{\text{PSMC}}(\rho, t_i, [lb_0, ub_K])}
$$

The PSMC probability combines a "no recombination" point mass with a "recombine and re-coalesce" integral:

$$
P_{\text{PSMC}}(\rho, s, [t_1, t_2]) = \mathbb{1}[t_1 \leq s \leq t_2] \cdot e^{-\rho \ell} + \int_{t_1}^{t_2} f_{\text{PSMC}}(\rho, s, t)\,dt
$$

where $\ell = 2s - \text{lower\_bound} - \text{cut\_time}$. Write `_psmc_cdf(rho, s, t)` to compute the CDF, then `_psmc_prob(rho, s, t1, t2) = base + psmc_cdf(t2) - psmc_cdf(t1)` where `base` is the point mass.

**Lower off-diagonal** (transition from interval $i+1$ down to $i$):

$$
L_i = \frac{P_{\text{PSMC}}(\rho, t_{i+1}, [lb_i, ub_i])}{P_{\text{PSMC}}(\rho, t_{i+1}, [lb_0, ub_K])}
$$

**Upper off-diagonal** (transition from interval $i-1$ up to $i$):

$$
U_i = \frac{P_{\text{PSMC}}(\rho, t_{i-1}, [lb_i, ub_i])}{P_{\text{PSMC}}(\rho, t_{i-1}, [lb_0, ub_K])}
$$

The key insight for efficiency is the **$O(K)$ recursion** using cumulative sums:

```python
# lower_sums[i] = sum over j<i of (U_j contribution)
lower_sums[0] = 0
for i in range(1, K):
    lower_sums[i] = upper_diagonals[i] * fp[i-1] + factors[i] * lower_sums[i-1]

# upper_sums[i] = sum of fp[i+1:]
upper_sums[K-1] = 0
for i in range(K-2, -1, -1):
    upper_sums[i] = fp[i+1] + upper_sums[i+1]
```

where `factors[i]` is the ratio of exponential masses between adjacent intervals: $(e^{-lb_i} - e^{-ub_i}) / (e^{-lb_{i-1}} - e^{-ub_{i-1}})$, clamped to at most 5.

The forward update is then:

$$
\alpha_i(x+1) = \text{lower\_sums}[i] + D_i \cdot \alpha_i(x) + L_i \cdot \text{upper\_sums}[i]
$$

### `transfer(r, prev_branch, next_branch)` — topology change

Three cases:

1. **Source → merging**: collapse to a point mass at `deleted_node.time`. Generate intervals on the merging branch with a point interval at that time.
2. **Target → recombined**: expand from a point mass. Generate intervals on the recombined branch, with full probability assigned above `start_time`.
3. **Regular transfer**: overlap intervals by time between the old and new branches, transferring mass proportionally.

### `recombine(prev_branch, next_branch)` — branch change without Recombination record

When the BSP sampled a branch change at a position that doesn't correspond to a topology change, generate a fresh grid on the new branch and redistribute mass using `_recomb_prob(prev_time, new_lb, new_ub)`.

### `sample_joining_nodes(start_index, coordinates)` — traceback

Walk backward, sampling an interval at each step. Convert each sampled interval to a `Node` at a time drawn from `_exp_median(lb, ub)` — a jittered exponential median that samples uniformly in quantile space:

$$
mq = (0.45 + 0.1 \cdot U) \cdot (uq - lq) + lq, \quad t = -\log(1 - mq)
$$

where $U \sim \text{Uniform}(0,1)$. This ensures diversity across MCMC iterations.

If the interval has a `.node` pointer (from a transfer point mass), reuse that node instead of creating a new one.

> **Code:** `pysinger/hmm/tsp.py`

---

## Step 11 — The Threader

**Goal.** Combine BSP + TSP into a single threading operation.

### `thread(arg, node)` — initial threading

This is used during `iterative_start()` to add a new leaf node:

```python
def thread(self, arg, node):
    arg.add_sample(node)           # 1. register node
    self._run_bsp(arg)             # 2. BSP forward pass
    self._sample_joining_branches(arg)  # 3. BSP traceback → pos → Branch
    self._run_tsp(arg)             # 4. TSP forward pass
    self._sample_joining_points(arg)    # 5. TSP traceback → pos → Node
    arg.add(joining_branches, added_branches)  # 6. thread into ARG
    arg.approx_sample_recombinations()  # 7. set recombination times
    arg.clear_remove_info()         # 8. clean up
```

### `_run_bsp(arg)` — the BSP forward pass loop

Iterate over the HMM grid from `start_index` to `end_index`:

```python
for i in range(start_index, end_index):
    pos = arg.coordinates[i]

    # Which query node is active at this position?
    query_node = arg.get_query_node_at(pos)

    # Transition: recombination transfer or regular forward step
    if pos is a recombination breakpoint:
        bsp.transfer(recombination)
    elif pos != start:
        bsp.forward(arg.rhos[i-1])

    # Collect mutations in [pos, next_pos)
    mut_set = {m for m in mutation_sites if pos ≤ m < next_pos}

    # Emission
    if mut_set:
        bsp.mut_emit(theta, bin_size, mut_set, query_node)
    else:
        bsp.null_emit(theta, query_node)
```

### `_run_tsp(arg)` — the TSP forward pass loop

Same structure, but conditioned on the BSP's sampled joining branches. The TSP uses `BinaryEmission` and handles three transition types: transfer (at recombination), recombine (branch change without topology change), and forward (same branch).

### `internal_rethread(arg, cut_point)` — MCMC move

```python
def internal_rethread(self, arg, cut_point):
    arg.remove(cut_point)          # 1. extract lineage
    self._run_bsp(arg)             # 2. propose new branch path
    self._sample_joining_branches(arg)
    self._run_tsp(arg)             # 3. propose new times
    self._sample_joining_points(arg)
    ar = self._acceptance_ratio(arg)  # 4. Metropolis ratio
    if random() < ar:
        arg.add(new_joining, new_added)     # accept
    else:
        arg.add(old_joining, old_removed)   # reject → restore
    arg.approx_sample_recombinations()
    arg.clear_remove_info()
```

### `_acceptance_ratio(arg)` — Metropolis criterion

$$
\alpha = \frac{h_{\text{old}}}{h_{\text{new}}}
$$

where $h$ is the **effective tree height** at the cut position. Compute it as the maximum child-node time in the cut tree. If the joining branch reaches the root sentinel, replace $h$ with the coalescence node's time. This ratio favours proposals that don't inflate tree heights.

### Emission model assignment

Use `PolarEmission` for the BSP (benefits from ancestral/derived polarity information) and `BinaryEmission` for the TSP (symmetric model suffices for time resolution).

> **Code:** `pysinger/mcmc/threader.py`

---

## Step 12 — The Sampler

**Goal.** Top-level MCMC orchestrator.

### Constructor

Store `Ne`, `recomb_rate × Ne`, `mut_rate × Ne`, precision parameters (`bsp_c`, `tsp_q`), and optionally variable-rate maps.

### `load_vcf(vcf_file, start, end, haploid)`

Parse a phased VCF into a list of `Node` objects (2 per diploid individual). Each node's `mutation_sites` is populated with positions carrying the derived allele. Shuffle the threading order deterministically using a seeded RNG.

### `iterative_start(max_retries=5)` — build initial ARG

```python
def iterative_start(self):
    self._build_singleton_arg()           # ARG with first sample
    for node in ordered_samples[1:]:      # thread remaining samples
        threader = self._make_threader()
        threader.thread(self.arg, node)
    self._rescale()                       # calibrate to mutation rate
```

Wrap in retry logic: if any threading step fails due to HMM underflow (the forward probabilities collapse to zero, which can happen on long sequences with sparse mutations), retry with a different RNG seed.

### `internal_sample(num_iters, spacing)` — MCMC loop

For each iteration, keep proposing re-threading moves until at least `spacing × sequence_length` base pairs have been updated:

```python
for iteration in range(num_iters):
    updated_length = 0
    while updated_length < spacing * sequence_length:
        cut_point = arg.sample_internal_cut()
        threader.internal_rethread(arg, cut_point)
        updated_length += (end - start)  # region affected
    self._rescale()
```

### `_rescale()` — mutation-rate calibration

After each iteration, rescale all internal node times so the expected number of mutations matches the observed count:

$$
s = \frac{S_{\text{obs}}}{\mu_{\text{scaled}} \cdot L_{\text{total}}}
$$

where $S_{\text{obs}}$ = number of unique segregating site positions across all samples, $\mu_{\text{scaled}} = \mu \cdot N_e$, and $L_{\text{total}} = \text{arg.get\_arg\_length()}$.

Collect all internal nodes by walking the ARG (they're not tracked in a set — discover them by replaying trees). Multiply each internal node's `time` by $s$, and also rescale each recombination's `start_time`.

> **Code:** `pysinger/sampler.py`

---

## Step 13 — VCF reader

**Goal.** Parse phased genotypes into `Node` objects.

Write `read_vcf_phased(path, start, end)`:

1. Parse the `#CHROM` header to count individuals. Create `2 × n_individuals` Node objects with `time = 0` and indices `0, 1, …`.
2. For each variant line: skip if outside `[start, end)`, skip duplicate positions and indels.
3. Parse the `0|1` genotype format. For each haplotype carrying the derived allele (`1`), call `node.add_mutation(pos - start)` (positions are stored relative to `start`).
4. Only include segregating sites: skip positions where all or no haplotypes carry the derived allele.
5. Return `(nodes, sequence_length)`.

Also write `read_vcf_haploid` for haploid data (one node per column).

> **Code:** `pysinger/io/vcf_reader.py`

---

## Step 14 — Fitch parsimony reconstruction

**Goal.** Assign ancestral states to internal nodes with minimum mutations.

Write `FitchReconstruction(tree)`:

### `reconstruct(pos)` — two-pass Fitch algorithm

**Pass 1 — Pruning (bottom-up):** For each node, compute its state from its children:

```python
if both children agree:     parent_state = that state
elif one is ambiguous (0.5): parent_state = the other's state
elif they disagree:          parent_state = 0.5 (ambiguous)
```

Leaf nodes read their state from `node.get_state(pos)`.

**Pass 2 — Peeling (top-down):** Resolve ambiguities:

```python
if parent is root sentinel:  state = 0 if ambiguous else pruning_state
elif parent has definite state: state = parent_state if ambiguous else pruning_state
```

After both passes, call `node.write_state(pos, resolved_state)` for every internal node.

### `update(recombination)`

Advance the tree by `tree.forward_update(r)`, then rebuild the parent/children/node_set dictionaries from the updated tree.

> **Code:** `pysinger/reconstruction/fitch.py`

---

## Step 15 — tskit export

**Goal.** Convert the final ARG to a `tskit.TreeSequence`.

Write `arg_to_tskit(arg, Ne)`:

1. **Discover nodes**: replay all recombinations, collecting every node that appears in a parent/child pair (excluding the root sentinel).
2. **Add nodes**: `tables.nodes.add_row(time=node.time * Ne, flags=NODE_IS_SAMPLE if sample)`.
3. **Emit edges**: replay again; for each tree interval $[x_k, x_{k+1})$, emit `tables.edges.add_row(left, right, parent_tskit_id, child_tskit_id)` for every parent-child pair where `parent.time > child.time` and parent isn't the root sentinel.
4. **Sort and build**: `tables.sort()` then `tables.tree_sequence()`.

> **Code:** `pysinger/io/tskit_writer.py`

---

## Putting it all together

Once you have all 15 pieces, the usage is:

```python
from pysinger import Sampler
from pysinger.io.tskit_writer import arg_to_tskit

sampler = Sampler(Ne=10000, recomb_rate=1e-8, mut_rate=1e-8)
sampler.load_vcf("data.vcf", start=0, end=1_000_000)
sampler.iterative_start()
sampler.internal_sample(num_iters=100, spacing=1)
ts = arg_to_tskit(sampler.arg, Ne=10000)
```

The dependency graph of the 15 steps is:

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
