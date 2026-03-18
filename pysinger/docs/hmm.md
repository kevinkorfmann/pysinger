# Hidden Markov Models

The HMM module (`pysinger.hmm`) contains the two coupled forward HMMs that are the computational heart of SINGER, plus the coalescent calculator and emission models that feed them.

## CoalescentCalculator

```{eval-rst}
.. autoclass:: pysinger.hmm.coalescent.CoalescentCalculator
   :members:
```

Computes the **piecewise-exponential coalescent CDF** for a set of branches. The coalescence rate at time $t$ equals the number of lineages alive at $t$:

$$
\lambda(t) = \sum_b \mathbb{1}[t_{\text{lower}}(b) \leq t < t_{\text{upper}}(b)]
$$

### Construction (`compute`)

1. **Rate changes**: Record $+1$ at each branch's lower time and $-1$ at each upper time.
2. **Cumulative rates**: Running sum gives the piecewise-constant rate function $\lambda(t)$.
3. **CDF**: The survival probability $S(t) = \exp(-\int_0^t \lambda(s)\,ds)$ is piecewise-exponential. Between consecutive rate-change times $[t_k, t_{k+1})$ with rate $\lambda_k$:

$$
S(t_{k+1}) = S(t_k) \cdot e^{-\lambda_k (t_{k+1} - t_k)}
$$

The cumulative probability $F(t) = 1 - S(t)$ is stored as a `SortedDict` for $O(\log n)$ lookup.

### Interpolation (`prob`)

For a time $t$ between grid points $[t_k, t_{k+1})$:

$$
F(t) = F(t_k) + \Delta F \cdot \frac{\text{expm1}(-\lambda_k \cdot (t - t_k))}{\text{expm1}(-\lambda_k \cdot (t_{k+1} - t_k))}
$$

This is exact (not an approximation) because the rate is constant within the interval.

### Quantile (`quantile`)

The inverse CDF: given $p$, find $t$ such that $F(t) = p$. Uses the inverse formula:

$$
t = t_k - \frac{1}{\lambda_k} \log\!\Bigl(1 - \frac{p - F(t_k)}{F(t_{k+1}) - F(t_k)} \cdot (1 - e^{-\lambda_k \Delta t})\Bigr)
$$

### `weight(lb, ub)` and `time(lb, ub)`

- `weight(lb, ub)` $= F(ub) - F(lb)$: probability of coalescence in $[lb, ub]$.
- `time(lb, ub)`: the exponential median -- the time $t$ at which $F(t)$ is midway between $F(lb)$ and $F(ub)$. Used as the representative time for each HMM interval.

## Emission models

```{eval-rst}
.. autoclass:: pysinger.hmm.emission.BinaryEmission
   :members:
.. autoclass:: pysinger.hmm.emission.PolarEmission
   :members:
```

Both models compute the **ratio** of the data likelihood with vs. without the threaded lineage. This ratio is the emission probability used by the HMM.

### BinaryEmission (used by TSP)

For a branch $b$ being split at time $t$ by a new lineage with query node $q$, the three sub-branch lengths are:

$$
\ell_{\text{lower}} = t - t_{\text{lower}}, \quad
\ell_{\text{upper}} = t_{\text{upper}} - t, \quad
\ell_{\text{query}} = t - t_q
$$

**Null emission** (no mutations in the bin):

$$
e_{\text{null}} = \frac{e^{-\theta \ell_l} \cdot e^{-\theta \ell_u} \cdot e^{-\theta \ell_q}}{e^{-\theta(\ell_l + \ell_u)}} = e^{-\theta \ell_q}
$$

**Mutation emission**: For each segregating site, compute the majority-rule state $s_m = \mathbb{1}[s_l + s_u + s_q > 1.5]$ and count transitions on each sub-branch. The probability is a product of Poisson terms:

$$
e_{\text{mut}} = \prod_{\text{sites}} \frac{e^{-\theta \ell_l} (\theta/\Delta x)^{d_l} \cdot e^{-\theta \ell_u} (\theta/\Delta x)^{d_u} \cdot e^{-\theta \ell_q} (\theta/\Delta x)^{d_q}}{e^{-\theta(\ell_l+\ell_u)} (\theta/\Delta x)^{d_{\text{old}}}}
$$

### PolarEmission (used by BSP)

Extends BinaryEmission with:
- **Penalty factor** for derived alleles in the query node (configurable via `penalty`).
- **Root reward**: when the branch reaches the root and the majority state is derived while the lower node is ancestral, applies a factor of `ancestral_prob / (1 - ancestral_prob)`.
- Simplified null emission: only depends on $\ell_q$ (finite branch) or $\ell_l + \ell_q$ (root branch).

## BSP — Branch Sequence Propagator

```{eval-rst}
.. autoclass:: pysinger.hmm.bsp.BSP
   :members:
```

The BSP is the **branch-dimension HMM**. Its state space is a list of `Interval` objects, one per (branch, time-range) cell in the current marginal tree.

### Forward pass

For each genomic bin $i$:

1. **Recombination or advance**: If position $i$ coincides with a recombination breakpoint, call `transfer(r)`. Otherwise call `forward(rho)`.
2. **Emission**: If mutations are present, call `mut_emit(theta, bin_size, mutations, query_node)`. Otherwise call `null_emit(theta, query_node)`.

### `forward(rho)` — no topology change

$$
\alpha_i^{(x+1)} = \alpha_i^{(x)} \cdot (1 - p_{\text{recomb},i}) + R \cdot w_i
$$

where $p_{\text{recomb},i} = \rho \cdot \Delta t_i \cdot e^{-\rho \Delta t_i}$ is the recombination probability for interval $i$ with $\Delta t_i = t_i - t_c$, and $w_i$ is the coalescent re-landing weight (coalescent probability $\times$ recombination probability, normalised).

### `transfer(r)` — topology change

Probability mass is redistributed across the new state space. The BSP classifies each old interval by its branch type (source / target / other) and routes mass to the corresponding new intervals. The `_generate_intervals` method builds the new state space, applying a pruning cutoff to discard low-probability partial intervals.

### Traceback

`sample_joining_branches()` walks backward through the forward probabilities, at each step deciding whether the lineage stayed on the same branch or jumped. Jumps are sampled proportional to the recombination probability $\times$ forward probability. The result is a `SortedDict[position -> Branch]`.

## TSP — Time Sequence Propagator

```{eval-rst}
.. autoclass:: pysinger.hmm.tsp.TSP
   :members:
```

The TSP is the **time-dimension HMM**. Conditioned on the BSP's sampled branch path, it samples a coalescence time for each genomic bin.

### State space

For a branch $[l, u]$, the time axis is discretised into $K$ intervals using **exponential-quantile spacing**: boundaries are placed at $F_{\text{Exp}(1)}^{-1}(p)$ for evenly-spaced $p$ values. This concentrates grid points where the coalescent density is highest.

The gap parameter $q$ (default 0.02) controls resolution: $K = \lceil (p_u - p_l) / q \rceil$.

### Forward pass

The PSMC transition kernel is a tridiagonal matrix:
- **Diagonal**: $P_{\text{stay}} = P_{\text{PSMC}}(\rho, t_i, [l_i, u_i]) / P_{\text{PSMC}}(\rho, t_i, [l_0, u_K])$
- **Lower diagonal**: probability of transitioning from interval $i+1$ to $i$
- **Upper diagonal**: probability of transitioning from interval $i-1$ to $i$

The $O(K)$ recursion uses `lower_sums` (cumulative sum from below) and `upper_sums` (cumulative sum from above):

$$
\alpha_i^{(x+1)} = L_i + D_i \cdot \alpha_i^{(x)} + U_i \cdot \sum_{j>i} \alpha_j^{(x)}
$$

### Transfer at branch changes

Three cases mirror the BSP:
- **Source $\to$ merging**: Collapse to a point mass at the deleted node's time, with intervals above and below.
- **Target $\to$ recombined**: Expand from a point mass, distributing probability across the new branch's time range.
- **Regular**: Transfer intervals by time overlap between old and new branches.

### Traceback

`sample_joining_nodes()` walks backward, sampling an `Interval` at each step and converting it to a `Node` at the interval's representative time (exponential median with jitter via `_exp_median`).
