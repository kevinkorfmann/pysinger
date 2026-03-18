# Implementation guide: building SINGER from scratch

This chapter is a step-by-step walkthrough for anyone who wants to **reimplement SINGER themselves** — in any language. Each step builds on the previous one, introduces exactly one new concept, and ends with a testable checkpoint. The order matches the dependency graph of the pysinger codebase.

---

## Step 0: Understand the goal

You are building a Bayesian sampler that takes a phased VCF and infers an **Ancestral Recombination Graph (ARG)** — a data structure that encodes the full genealogical history (with recombination) of a set of haplotypes.

The core loop is:
1. Thread haplotypes one-by-one into a growing ARG (initialisation).
2. Repeatedly pick a random lineage, remove it, and re-thread it (MCMC).

Threading requires two coupled HMMs: one to choose **which branch** to join (BSP), one to choose **when** to join (TSP).

**Prerequisite knowledge**: coalescent theory, hidden Markov models, Metropolis--Hastings MCMC.

---

## Step 1: Node and Branch

**What to build**: The two atomic types — vertices and directed edges.

**Key design decisions**:
- Nodes use **identity-based equality** (pointer/reference comparison), not value equality. Two nodes at the same time are still different objects.
- Branches are **immutable** and hashable (they live in sets and as dict keys).
- A special "null branch" (`Branch()` with both nodes `None`) serves as a sentinel.

**Files**: `data/node.py`, `data/branch.py`

**Checkpoint**: Create a few nodes and branches. Verify that `Branch(a, b) == Branch(a, b)` but `Branch(a, b) != Branch(a, c)`. Verify branches can be added to sets and used as dict keys.

```python
a = Node(time=0.0, index=0)
b = Node(time=1.0, index=1)
c = Node(time=1.0, index=2)
assert Branch(a, b) == Branch(a, b)    # same node objects → equal
assert Branch(a, b) != Branch(a, c)    # different upper node → not equal
assert len({Branch(a, b), Branch(a, b)}) == 1  # deduplicates in sets
```

---

## Step 2: Tree

**What to build**: A marginal coalescent tree — the tree at a single genomic position.

**Key design decisions**:
- Stored as two dicts: `parents[child] = parent` and `children[parent] = {child1, child2}`.
- Topology changes are applied via `forward_update(r)` (delete old branches, insert new ones) and `backward_update(r)` (reverse).
- `find_joining_branch(removed_branch)` returns the `Branch(sibling, grandparent)` that takes over when a coalescence node is pruned — this is the key helper for MCMC.

**Files**: `data/tree.py`

**Checkpoint**: Build a 3-leaf tree manually. Verify `forward_update` + `backward_update` is the identity. Verify `find_joining_branch` returns the correct sibling-to-grandparent branch.

---

## Step 3: Recombination

**What to build**: A topology-change record at a single genomic breakpoint.

**Key design decisions**:
- Stores `deleted_branches` (exist left of breakpoint) and `inserted_branches` (exist right of breakpoint).
- From these, derive named branches: `source_branch`, `target_branch`, `merging_branch`, `recombined_branch`, `lower/upper_transfer_branch`.
- `trace_forward(t, branch)` and `trace_backward(t, branch)` map a branch at height $t$ across the topology change — essential for `ARG.remove()`.

**Files**: `data/recombination.py`

**Checkpoint**: Create a simple SPR recombination by hand. Verify that applying `forward_update` to a tree produces the correct new topology. Verify `trace_forward` correctly maps branches across the event.

**This is the hardest data structure.** Take your time here. The derived fields (`source_branch`, `merging_branch`, etc.) are computed by `_find_nodes()`, `_find_target_branch()`, and `_find_recomb_info()`. Get these right and the HMM transfer steps will follow naturally.

---

## Step 4: Interval and IntervalInfo

**What to build**: Time-height intervals on branches — the state-space cells for the HMMs.

**Key design decisions**:
- `Interval(branch, lb, ub, start_pos)` represents a (branch, time-range) cell.
- `fill_time()` sets the representative time to the **exponential median**: $t^* = -\log(1 - \frac{1}{2}[(1-e^{-lb}) + (1-e^{-ub})])$.
- `IntervalInfo(branch, lb, ub)` is a lightweight hashable key for accumulating transfer weights.

**Files**: `data/interval.py`

**Checkpoint**: Create intervals on a branch. Verify `fill_time()` gives a time between `lb` and `ub` that is biased toward the lower end (where coalescent density is highest).

---

## Step 5: ARG (skeleton)

**What to build**: The central data structure — a sorted map of Recombination records.

**Start with just these methods**:
- `__init__`: Create root sentinel, recombination sentinels at 0 and $\infty$.
- `build_singleton_arg(node)`: An ARG with one sample connected to root.
- `get_tree_at(x)`: Replay recombinations up to position $x$.
- `discretize(bin_size)`: Build the HMM coordinate grid.
- `compute_rhos_thetas(r, m)`: Compute per-bin rates.

**Defer for later**: `remove()`, `add()`, `approx_sample_recombinations()`, `sample_internal_cut()`.

**Files**: `data/arg.py`

**Checkpoint**: Build a singleton ARG. Verify `get_tree_at(0)` returns a tree with one branch (sample → root). Verify `discretize` produces a grid covering $[0, L]$.

---

## Step 6: VCF reader

**What to build**: Parse a phased VCF into a list of `Node` objects with mutation data.

**Files**: `io/vcf_reader.py`

**Checkpoint**: Write a tiny VCF by hand. Parse it. Verify each node has the correct `mutation_sites` entries.

---

## Step 7: CoalescentCalculator

**What to build**: The piecewise-exponential coalescent CDF.

**The math**: Given $m$ branches alive at time $t$, the coalescence rate is $m$. The CDF $F(t) = 1 - \exp(-\int_0^t m(s)\,ds)$ is piecewise-exponential.

**Key methods**:
- `compute(branches)`: Build rate changes → cumulative rates → CDF.
- `weight(lb, ub)`: $F(ub) - F(lb)$.
- `time(lb, ub)`: Exponential median (representative time).
- `prob(x)`: CDF at arbitrary $x$ (interpolation).
- `quantile(p)`: Inverse CDF.

**Files**: `hmm/coalescent.py`

**Checkpoint**: For a single branch $[0, \infty)$, `weight(0, t)` should equal $1 - e^{-t}$ (standard exponential CDF). For two branches overlapping at $[0, 1)$ and $[0, \infty)$, the rate is 2 in $[0, 1)$ and 1 in $[1, \infty)$ — verify the CDF matches.

---

## Step 8: Emission models

**What to build**: The likelihood ratio of observing mutation data with vs. without the threaded lineage.

**Two models**:
- `BinaryEmission`: Symmetric infinite-sites model. Used by the TSP.
- `PolarEmission`: Ancestral/derived model with penalty and root reward. Used by the BSP.

**The key formula** (null emission, BinaryEmission):

$$
e = \frac{e^{-\theta \ell_l} \cdot e^{-\theta \ell_u} \cdot e^{-\theta \ell_q}}{e^{-\theta(\ell_l + \ell_u)}} = e^{-\theta \ell_q}
$$

**Files**: `hmm/emission.py`

**Checkpoint**: For a branch from time 0 to time 2, splitting at time 1 with a query at time 0 and $\theta = 0.1$: null emission should be $e^{-0.1 \times 1} = e^{-0.1} \approx 0.905$.

---

## Step 9: BSP (Branch Sequence Propagator)

**What to build**: The forward HMM over branches. This is the largest single component.

**Build in this order**:
1. `start()`: Initialise from a tree's branches using `CoalescentCalculator`.
2. `forward(rho)`: One-bin transition (stay + recombine).
3. `null_emit()` / `mut_emit()`: Apply emission, normalise.
4. `transfer(r)`: Redistribute mass at a topology change. Three cases: source, target, other.
5. `sample_joining_branches()`: Stochastic traceback.

**Files**: `hmm/bsp.py`

**Checkpoint**: Thread a second haplotype into a singleton ARG. Run BSP forward. Verify that forward probabilities sum to ~1 at each step. Run traceback and verify the sampled branch path makes sense (branches should exist in the corresponding trees).

---

## Step 10: TSP (Time Sequence Propagator)

**What to build**: The forward HMM over coalescence times on a single branch.

**Build in this order**:
1. `_generate_grid()` / `_generate_intervals()`: Quantile-spaced time grid.
2. `start()`: Initialise with exponential prior.
3. `forward(rho)`: PSMC transition (tridiagonal matrix).
4. `null_emit()` / `mut_emit()`: Apply emission, normalise.
5. `transfer()`: Handle branch switches (source→merging, target→recombined, regular).
6. `recombine()`: Full re-sample when the BSP switches branches without a topology change.
7. `sample_joining_nodes()`: Stochastic traceback → `Dict[pos, Node]`.

**Key insight**: The PSMC kernel's tridiagonal structure lets you compute the forward step in $O(K)$ instead of $O(K^2)$. The `lower_sums` / `upper_sums` / `factors` arrays implement this.

**Files**: `hmm/tsp.py`

**Checkpoint**: Run TSP on a single branch with no topology changes. The sampled coalescence times should fall within the branch's time range and be roughly exponentially distributed.

---

## Step 11: Threader

**What to build**: The glue that runs BSP + TSP in sequence to produce a complete threading.

**Two entry points**:
- `thread(arg, node)`: Add a new leaf (used during initialisation).
- `internal_rethread(arg, cut_point)`: Remove + re-thread with Metropolis acceptance (used during MCMC).

**Files**: `mcmc/threader.py`

**Checkpoint**: Thread 3 haplotypes into an ARG. Verify all sample nodes are connected in the tree at every position. Verify `arg.get_tree_at(x).parents` has the right number of entries ($2n - 1$ for $n$ samples, including the root edge).

---

## Step 12: ARG remove/add

**What to build**: The MCMC's core operations — extracting and re-inserting a lineage.

**`remove(cut_point)`**:
1. Trace the cut branch forward through recombinations (modifying each record).
2. Trace backward similarly.
3. Record `removed_branches` and `joining_branches`.
4. Update the cut tree.

**`add(joining_branches, added_branches)`**:
1. Walk through added positions.
2. At each: modify existing recombination or create new one.
3. Impute mutation states for new nodes.

Also implement: `approx_sample_recombinations()`, `sample_internal_cut()`, `get_check_points()`.

**Files**: `data/arg.py` (complete the skeleton from Step 5)

**Checkpoint**: Remove a lineage and immediately add it back unchanged. The ARG should be identical before and after. Then do a real re-thread via the Threader — the ARG should still produce valid trees at every position.

---

## Step 13: Sampler

**What to build**: The top-level orchestrator.

**Key methods**:
- `load_vcf()`: Parse VCF → nodes.
- `iterative_start()`: Thread all samples (with retry logic for underflow).
- `internal_sample(num_iters)`: MCMC loop with error recovery and rescaling.
- `_rescale()`: Global scale factor $s = S_{\text{obs}} / (\mu_{\text{scaled}} \cdot L_{\text{total}})$.

**Files**: `sampler.py`

**Checkpoint**: Run the full pipeline on a small simulated VCF. Verify the inferred ARG has 0 multi-root trees when exported to tskit. Run 100+ MCMC iterations and verify convergence diagnostics stabilise.

---

## Step 14: tskit export

**What to build**: Convert the pysinger ARG to a `tskit.TreeSequence`.

**Algorithm**:
1. Walk all trees to discover every node (except root sentinel).
2. Add nodes to tskit tables (`time = node.time * Ne`).
3. Walk all trees again, emitting edges for each parent-child pair.
4. `tables.sort()` → `tables.tree_sequence()`.

**Files**: `io/tskit_writer.py`

**Checkpoint**: Export and verify with `ts.trees()`. Every tree should be single-root. TMRCAs should be in the right ballpark compared to truth (if using simulated data).

---

## Step 15: Fitch parsimony (optional)

**What to build**: Ancestral state reconstruction for mutation placement.

**Already used implicitly** by the emission models and `ARG._impute()`, but having a standalone `FitchReconstruction` class is useful for debugging and for re-reconstructing states after topology changes.

**Files**: `reconstruction/fitch.py`

---

## Step 16: Polish and validate

With all pieces in place:

1. **Convergence diagnostics**: Track breakpoint count, incompatible sites, mean TMRCA.
2. **Scaling to longer sequences**: Ensure no HMM underflow (add retry logic).
3. **Comparison with truth**: Simulate with msprime/stdpopsim, infer with pysinger, compare TMRCA distributions and breakpoint locations.
4. **Performance**: Profile and optimise hot loops (BSP forward, TSP forward, emission computation).

---

## Dependency graph

```
Step 1: Node, Branch
  │
Step 2: Tree
  │
Step 3: Recombination
  │
Step 4: Interval, IntervalInfo
  │
Step 5: ARG (skeleton)          Step 6: VCF reader
  │                               │
Step 7: CoalescentCalculator      │
  │                               │
Step 8: Emission models           │
  │                               │
Step 9: BSP ─────────────────────┐│
  │                              ││
Step 10: TSP ────────────────────┘│
  │                               │
Step 11: Threader                 │
  │                               │
Step 12: ARG (remove/add)        │
  │                               │
Step 13: Sampler ─────────────────┘
  │
Step 14: tskit export
  │
Step 15: Fitch parsimony
  │
Step 16: Validate
```

Each step depends only on the steps above it. You can write tests at each checkpoint before moving on.
