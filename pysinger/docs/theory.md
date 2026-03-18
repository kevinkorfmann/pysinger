# Mathematical background

This chapter derives the key equations implemented in pysinger. All notation follows the SINGER paper (Deng, Nielsen & Song, *Nat Genet* 2025).

## 1. The coalescent with recombination

An **Ancestral Recombination Graph (ARG)** encodes the genealogical history of a sample of $n$ haplotypes over a genomic region $[0, L)$. At each position $x$ the ARG induces a **marginal coalescent tree** $\mathcal{T}(x)$. Recombination events at positions $x_1, x_2, \ldots$ change the tree topology:

$$
\mathcal{T}(0) \xrightarrow{x_1} \mathcal{T}_1 \xrightarrow{x_2} \mathcal{T}_2 \xrightarrow{x_3} \cdots
$$

Under the **Sequentially Markov Coalescent (SMC)**, adjacent trees differ by exactly one **subtree-prune-and-regraft (SPR)** operation: one lineage detaches at the recombination breakpoint and re-coalesces with a different branch at a new time.

### Coalescent rate

Given $k$ lineages at time $t$ in a population of effective size $N_e$, the rate at which any pair coalesces is:

$$
\lambda(k) = \binom{k}{2} \cdot \frac{1}{2N_e} \quad \text{per generation}
$$

For a **new lineage** being threaded into an existing tree with $m$ branches spanning time $t$, the coalescence rate is simply $m$ (in coalescent units where $1\text{ unit} = N_e$ generations):

$$
\lambda(t) = m(t) = \#\{\text{branches alive at time } t\}
$$

This is computed by `CoalescentCalculator`. It records $+1$ when a branch starts and $-1$ when it ends, giving a piecewise-constant rate function. The CDF of coalescence time is then:

$$
F(t) = 1 - \exp\!\Bigl(-\int_0^t m(s)\,ds\Bigr)
$$

which is piecewise-exponential and computed exactly by `CoalescentCalculator.prob()`.

## 2. The Branch Sequence Propagator (BSP)

The BSP is a forward HMM that answers: **which branch** should the new lineage join at each genomic position?

### State space

At position $x$, the state space is the set of **(branch, time-interval)** pairs in the current marginal tree above the cut time $t_c$:

$$
\mathcal{S}(x) = \{(b, [l_b, u_b]) : b \in \mathcal{T}(x),\; u_b > t_c\}
$$

Each state $i$ carries a forward probability $\alpha_i(x)$.

### Initialisation

At the left boundary, forward probabilities are set to the coalescent weights:

$$
\alpha_i(0) = \Pr(\text{coalesce in } [l_i, u_i]) = F(u_i) - F(l_i)
$$

### Transition (forward step)

Between positions $x$ and $x + \Delta x$ (no topology change), recombination occurs with probability depending on the recombination rate $\rho = r \cdot \Delta x$ and the branch time:

$$
p_{\text{recomb}}(i) = \rho \cdot (t_i - t_c) \cdot e^{-\rho(t_i - t_c)}
$$

The forward update mixes staying on the same branch with jumping to a new one:

$$
\alpha_i(x + \Delta x) = \alpha_i(x) \cdot (1 - p_{\text{recomb}}(i)) + R \cdot w_i
$$

where $R = \sum_j p_{\text{recomb}}(j) \cdot \alpha_j(x)$ is the total recombination mass and $w_i$ is the coalescent re-landing weight of state $i$.

### Transfer (at recombination breakpoint)

When the tree topology changes at a recombination position, the state space itself changes. The BSP maps probability mass from old states to new ones based on the branch correspondence defined by the `Recombination` record. Three cases arise:

- **Source branch** (the lineage that recombines): mass below the recombination height goes to the `recombined_branch`; mass above collapses to a point mass on the `merging_branch`.
- **Target branch** (where the lineage re-coalesces): mass is split between `lower_transfer_branch` and `upper_transfer_branch`, with some mass moving to the `recombined_branch` via `_get_overwrite_prob`.
- **Other branches**: mass transfers to the topologically equivalent branch in the new tree (identity or `merging_branch`).

### Emission

For a genomic bin with scaled mutation rate $\theta = \mu \cdot \Delta x$:

**No mutations** (null emission): The lineage splitting branch $b$ at time $t$ into a lower segment $(t - t_{\text{lower}})$, upper segment $(t_{\text{upper}} - t)$, and query segment $(t - t_{\text{query}})$ produces:

$$
P(\text{no mut}) = \frac{e^{-\theta \ell_{\text{lower}}} \cdot e^{-\theta \ell_{\text{upper}}} \cdot e^{-\theta \ell_{\text{query}}}}{e^{-\theta(\ell_{\text{lower}} + \ell_{\text{upper}})}}
= e^{-\theta \ell_{\text{query}}}
$$

This is the ratio of the new tree's no-mutation probability to the old tree's, cancelling shared terms. The BSP uses the `PolarEmission` model which also accounts for ancestral/derived polarisation.

**With mutations**: For each segregating site in the bin, the emission computes a majority-rule ancestral state $s_m = \mathbb{1}[s_l + s_u + s_0 > 1.5]$ and counts the number of state changes on each sub-branch:

$$
P(\text{mut}) \propto \prod_{\text{sites}} \frac{(\theta/\Delta x)^{d_l} \cdot (\theta/\Delta x)^{d_u} \cdot (\theta/\Delta x)^{d_q}}{(\theta/\Delta x)^{d_{\text{old}}}}
$$

where $d_l = |s_m - s_l|$, etc.

### Traceback

After the forward pass, the BSP samples a path of joining branches by stochastic traceback. At each step it decides whether to stay on the current branch or jump to a new one, weighted by the forward probabilities and recombination rates.

## 3. The Time Sequence Propagator (TSP)

Conditioned on the BSP's sampled joining branches, the TSP is a forward HMM that answers: **at what time** should the new lineage coalesce on each branch?

### State space

For a single branch $b = [\ell, u]$, the TSP discretises the time axis into quantile-spaced intervals using the $\text{Exp}(1)$ CDF:

$$
q_k = F_{\text{Exp}}^{-1}\!\bigl(p_l + k \cdot \Delta p\bigr), \quad k = 0, \ldots, K
$$

where $p_l = 1 - e^{-\ell}$, $p_u = 1 - e^{-u}$, and $\Delta p = (p_u - p_l)/K$ with $K = \lceil (p_u - p_l) / q \rceil$ for gap parameter $q$ (default 0.02). This spacing ensures finer resolution where the coalescent density is highest.

### Transition (PSMC kernel)

The TSP uses a **PSMC-style** transition kernel. Given current time $s$ and recombination rate $\rho$:

$$
P_{\text{PSMC}}(\rho, s, [t_1, t_2]) = \mathbb{1}[t_1 \leq s \leq t_2] \cdot e^{-\rho \ell} + \int_{t_1}^{t_2} f_{\text{PSMC}}(\rho, s, t)\,dt
$$

The first term is a point mass (no recombination: stay at the same time). The integral term accounts for recombination followed by re-coalescence at a new time $t$, computed via:

$$
\text{CDF}_{\text{PSMC}}(\rho, s, t) = \frac{1 - e^{-\rho \ell}}{\ell} \cdot \int_0^t g(s, u)\,du
$$

where $\ell = 2s - t_{\text{lower}} - t_c$ and $g$ depends on whether $t \leq s$ or $t > s$.

The transition matrix is **tridiagonal** (adjacent intervals interact most strongly), allowing efficient $O(K)$ forward updates via the `lower_sums` / `upper_sums` recursion.

### Transfer at topology changes

When the BSP switches branches at a recombination, the TSP handles three cases:
- **Source → merging**: collapse to a point mass at the deleted node's time.
- **Target → recombined**: expand from a point mass to the recombined branch's time range.
- **Regular transfer**: overlap intervals between old and new branches, preserving probability mass proportionally.

## 4. The MCMC

### Initialisation (`iterative_start`)

The initial ARG is built by threading haplotypes one by one:

1. Start with a singleton ARG (one sample connected to the root sentinel).
2. For each additional sample: run BSP → sample joining branches → run TSP → sample coalescence times → insert into ARG.
3. After all samples are threaded, rescale branch lengths.

### Metropolis--Hastings moves (`internal_sample`)

Each MCMC iteration:

1. **Sample a cut point** $(x_0, b, t_c)$: pick a random position, branch, and time in the current ARG.
2. **Remove** the lineage passing through $(b, t_c)$ by tracing it forward and backward through recombination records, yielding `removed_branches` and `joining_branches`.
3. **Propose** a new threading by running BSP + TSP on the modified ARG.
4. **Accept/reject** with Metropolis ratio:

$$
\alpha = \frac{h_{\text{old}}}{h_{\text{new}}}
$$

where $h$ is the height of the cut tree (the maximum coalescence time at the cut position, adjusted for root-reaching branches). This ratio favours proposals that don't unnecessarily inflate tree heights.

### Rescaling

After each MCMC iteration, a global rescaling adjusts all internal node times so that the expected number of mutations matches the observed count:

$$
s = \frac{S_{\text{obs}}}{\mu_{\text{scaled}} \cdot L_{\text{total}}}
$$

where $S_{\text{obs}}$ is the number of segregating sites, $\mu_{\text{scaled}} = \mu \cdot N_e$, and $L_{\text{total}} = \sum_{\text{trees}} (\text{branch length} \times \text{genomic span})$. All internal node times and recombination start times are multiplied by $s$.

## 5. Ancestral state reconstruction (Fitch parsimony)

After threading, mutations need to be placed on branches. pysinger uses **Fitch parsimony**:

**Up-pass (pruning)**: For each internal node, compute the intersection of children's states. If the intersection is empty (children disagree), mark as ambiguous (0.5).

$$
s_p = \begin{cases}
s_{c_1} & \text{if } s_{c_1} = s_{c_2} \\
s_{c_i} & \text{if the other child is ambiguous} \\
0.5 & \text{if } s_{c_1} \neq s_{c_2} \text{ (both definite)}
\end{cases}
$$

**Down-pass (peeling)**: Resolve ambiguities top-down. If a node is ambiguous, take the parent's resolved state. The root sentinel resolves ambiguity to the ancestral state (0).

## 6. ARG encoding

The ARG is stored as a **sorted map** of `Recombination` records keyed by genomic position. Each record stores the set of branches deleted (existing before the breakpoint) and inserted (existing after). The marginal tree at any position $x$ is obtained by replaying all records from position 0 up to $x$:

```python
tree = Tree()
for pos, r in arg.recombinations.items():
    if pos <= x:
        tree.forward_update(r)  # delete old branches, insert new ones
```

This "tape replay" representation is compact and supports efficient forward/backward traversal, which is critical for the BSP and MCMC remove/add operations.
