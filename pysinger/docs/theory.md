# Mathematical background

This chapter derives the key equations implemented in pysinger and shows the corresponding Python code side-by-side. All notation follows the SINGER paper (Deng, Nielsen & Song, *Nat Genet* 2025).

## 1. The coalescent with recombination

An **Ancestral Recombination Graph (ARG)** encodes the genealogical history of a sample of $n$ haplotypes over a genomic region $[0, L)$. At each position $x$ the ARG induces a **marginal coalescent tree** $\mathcal{T}(x)$. Recombination events at positions $x_1, x_2, \ldots$ change the tree topology:

$$
\mathcal{T}(0) \xrightarrow{x_1} \mathcal{T}_1 \xrightarrow{x_2} \mathcal{T}_2 \xrightarrow{x_3} \cdots
$$

Under the **Sequentially Markov Coalescent (SMC)**, adjacent trees differ by exactly one **subtree-prune-and-regraft (SPR)** operation: one lineage detaches at the recombination breakpoint and re-coalesces with a different branch at a new time.

In code, the ARG stores this sequence of topology changes as a sorted map of `Recombination` records. Retrieving the tree at any position replays the tape from left to right:

```python
# pysinger/data/arg.py — ARG.get_tree_at
def get_tree_at(self, x: float) -> Tree:
    tree = Tree()
    for pos, r in self.recombinations.items():
        if pos <= x:
            tree.forward_update(r)   # delete old branches, insert new
        else:
            break
    return tree
```

Each `forward_update` applies one SPR: it deletes the `r.deleted_branches` and inserts `r.inserted_branches` into the tree's parent/child dicts.

### Coalescent rate

Given $k$ lineages at time $t$ in a population of effective size $N_e$, the rate at which any pair coalesces is:

$$
\lambda(k) = \binom{k}{2} \cdot \frac{1}{2N_e} \quad \text{per generation}
$$

For a **new lineage** being threaded into an existing tree with $m$ branches spanning time $t$, the coalescence rate is simply $m$ (in coalescent units where $1\text{ unit} = N_e$ generations):

$$
\lambda(t) = m(t) = \#\{\text{branches alive at time } t\}
$$

`CoalescentCalculator` builds this piecewise-constant rate by scanning all branches and recording $+1$ at each branch's start time and $-1$ at its end:

```python
# pysinger/hmm/coalescent.py — CoalescentCalculator._compute_rate_changes
def _compute_rate_changes(self, branches):
    self._rate_changes = SortedDict()
    for b in branches:
        lb = max(self.cut_time, b.lower_node.time)
        ub = b.upper_node.time
        self._rate_changes[lb] = self._rate_changes.get(lb, 0) + 1
        self._rate_changes[ub] = self._rate_changes.get(ub, 0) - 1
```

A cumulative sum then gives the rate at every time point:

```python
# CoalescentCalculator._compute_rates
def _compute_rates(self):
    self._rates = SortedDict()
    curr = 0
    for t, delta in self._rate_changes.items():
        curr += delta
        self._rates[t] = curr
```

### Coalescent CDF

The CDF of coalescence time is:

$$
F(t) = 1 - \exp\!\Bigl(-\int_0^t m(s)\,ds\Bigr)
$$

Since $m(s)$ is piecewise constant, the integral is a sum of rectangles and the survival function $S(t) = 1 - F(t)$ is piecewise exponential. Between consecutive rate-change times $[t_k, t_{k+1})$ with rate $\lambda_k$:

$$
S(t_{k+1}) = S(t_k) \cdot e^{-\lambda_k \cdot (t_{k+1} - t_k)}
$$

```python
# CoalescentCalculator._compute_probs_quantiles
rate_keys = list(self._rates.keys())
prev_prob = 1.0    # S(t_0) = 1
cum_prob = 0.0     # F(t_0) = 0

for i in range(len(rate_keys) - 1):
    curr_rate = self._rates[rate_keys[i]]
    delta_t = rate_keys[i + 1] - rate_keys[i]

    if curr_rate > 0:
        next_prob = prev_prob * math.exp(-curr_rate * delta_t)
        cum_prob += prev_prob - next_prob   # F += S_old - S_new
    else:
        next_prob = prev_prob               # no coalescence possible

    self._cum_probs[rate_keys[i + 1]] = cum_prob
    prev_prob = next_prob
```

For times between grid points, the CDF is interpolated exactly using the exponential formula:

$$
F(t) = F(t_k) + \Delta F \cdot \frac{\text{expm1}(-\lambda_k \cdot (t - t_k))}{\text{expm1}(-\lambda_k \cdot \Delta t)}
$$

```python
# CoalescentCalculator.prob — interpolation for arbitrary t
denom = math.expm1(-rate * delta_t)
if abs(denom) < 1e-15:
    new_delta_p = delta_p * new_delta_t / delta_t       # linear fallback
else:
    new_delta_p = delta_p * math.expm1(-rate * new_delta_t) / denom
return base_prob + new_delta_p
```

The inverse (quantile function) uses the inverse formula:

$$
t = t_k - \frac{1}{\lambda_k} \log\!\Bigl(1 - \frac{p - F(t_k)}{F(t_{k+1}) - F(t_k)} \cdot (1 - e^{-\lambda_k \Delta t})\Bigr)
$$

```python
# CoalescentCalculator.quantile
frac = new_delta_p / delta_p * (1.0 - math.exp(-rate * delta_t))
new_delta_t = -math.log(1.0 - frac) / rate
return base_time + new_delta_t
```

## 2. The Branch Sequence Propagator (BSP)

The BSP is a forward HMM that answers: **which branch** should the new lineage join at each genomic position?

### State space

At position $x$, the state space is the set of **(branch, time-interval)** pairs in the current marginal tree above the cut time $t_c$:

$$
\mathcal{S}(x) = \{(b, [l_b, u_b]) : b \in \mathcal{T}(x),\; u_b > t_c\}
$$

Each state $i$ is an `Interval` object carrying a forward probability $\alpha_i(x)$.

### Initialisation

At the left boundary, forward probabilities are set to the coalescent weights — the probability that a new lineage coalesces within each interval:

$$
\alpha_i(0) = \Pr(\text{coalesce in } [l_i, u_i]) = F(u_i) - F(l_i)
$$

```python
# pysinger/hmm/bsp.py — BSP.start
self.cc = CoalescentCalculator(t)
self.cc.compute(self.valid_branches)

for b in sorted(self.valid_branches, key=lambda x: x):
    lb = max(b.lower_node.time, t)
    ub = b.upper_node.time
    p = self.cc.weight(lb, ub)          # F(ub) - F(lb)
    interval = Interval(b, lb, ub, self.curr_index)
    self.curr_intervals.append(interval)
    temp.append(p)

self.forward_probs.append(temp)
```

### Transition (forward step)

Between positions $x$ and $x + \Delta x$ (no topology change), recombination occurs with a probability that depends on the recombination rate $\rho = r \cdot \Delta x$ and the branch's representative time:

$$
p_{\text{recomb}}(i) = \rho \cdot (t_i - t_c) \cdot e^{-\rho(t_i - t_c)}
$$

```python
# BSP._get_recomb_prob
def _get_recomb_prob(self, rho, t):
    dt = t - self.cut_time
    return rho * dt * math.exp(-rho * dt)
```

The forward update mixes staying on the same branch with jumping to a new one:

$$
\alpha_i(x + \Delta x) = \alpha_i(x) \cdot (1 - p_{\text{recomb}}(i)) + R \cdot w_i
$$

where $R = \sum_j p_{\text{recomb}}(j) \cdot \alpha_j(x)$ is the total recombination mass and $w_i$ is the coalescent re-landing weight of state $i$.

```python
# BSP.forward
prev_fp = self.forward_probs[self.curr_index - 1]
self.recomb_sum = sum(
    self.recomb_probs[i] * prev_fp[i] for i in range(self.dim)
)

new_fp = [0.0] * self.dim
for i in range(self.dim):
    new_fp[i] = (
        prev_fp[i] * (1.0 - self.recomb_probs[i])      # stay
        + self.recomb_sum * self.recomb_weights[i]        # jump
    )
self.forward_probs.append(new_fp)
```

### Transfer (at recombination breakpoint)

When the tree topology changes at a recombination position, the state space itself changes. The BSP maps probability mass from old states to new ones based on the branch correspondence defined by the `Recombination` record. Three cases arise:

**Source branch** (the lineage that recombines): mass below the recombination height goes to `recombined_branch`; mass above collapses to a point mass on `merging_branch`.

```python
# BSP._process_source_interval
if prev.ub <= break_time:
    # entirely below recombination → recombined branch
    key = IntervalInfo(r.recombined_branch, prev.lb, prev.ub)
    self._transfer_helper(key, prev, p, tw, ti)
elif prev.lb >= break_time:
    # entirely above → point mass on merging branch
    key = IntervalInfo(r.merging_branch, point_time, point_time)
    self._transfer_helper(key, prev, p, tw, ti)
else:
    # straddles: split mass proportionally
    w1 = self.cc.weight(prev.lb, break_time)
    w2 = self.cc.weight(break_time, prev.ub)
    # ... normalise and route to recombined / merging
```

**Target branch** (where the lineage re-coalesces): mass is split between `lower_transfer_branch` and `upper_transfer_branch`, with some mass moving to `recombined_branch` via `_get_overwrite_prob`.

**Other branches**: mass transfers to the topologically equivalent branch in the new tree.

```python
# BSP._process_other_interval
if r.affect(prev.branch):
    key = IntervalInfo(r.merging_branch, prev.lb, prev.ub)
else:
    key = IntervalInfo(prev.branch, prev.lb, prev.ub)   # unchanged
self._transfer_helper(key, prev, p, tw, ti)
```

### Emission

For a genomic bin with scaled mutation rate $\theta = \mu \cdot \Delta x$:

**No mutations** (null emission): The lineage splitting branch $b$ at time $t$ into a lower segment $\ell_l = t - t_{\text{lower}}$, upper segment $\ell_u = t_{\text{upper}} - t$, and query segment $\ell_q = t - t_{\text{query}}$ produces:

$$
P(\text{no mut}) = \frac{e^{-\theta \ell_l} \cdot e^{-\theta \ell_u} \cdot e^{-\theta \ell_q}}{e^{-\theta(\ell_l + \ell_u)}}
= e^{-\theta \ell_q}
$$

This is the ratio of the new tree's no-mutation probability to the old tree's, cancelling shared terms.

```python
# pysinger/hmm/emission.py — BinaryEmission.null_emit
ll = time - branch.lower_node.time
lu = branch.upper_node.time - time
l0 = time - node.time
emit_prob = (
    self._calculate_prob(ll * theta, 1, 0)     # exp(-theta * ll)
    * self._calculate_prob(lu * theta, 1, 0)   # exp(-theta * lu)
    * self._calculate_prob(l0 * theta, 1, 0)   # exp(-theta * l0)
)
old_prob = self._calculate_prob((ll + lu) * theta, 1, 0)
return emit_prob / old_prob    # cancels to exp(-theta * l0)
```

**With mutations**: For each segregating site in the bin, the emission computes a majority-rule ancestral state $s_m = \mathbb{1}[s_l + s_u + s_0 > 1.5]$ and counts state changes on each sub-branch ($d_l = |s_m - s_l|$, etc.):

$$
P(\text{mut}) \propto \prod_{\text{sites}} \frac{(\theta/\Delta x)^{d_l} \cdot (\theta/\Delta x)^{d_u} \cdot (\theta/\Delta x)^{d_q}}{(\theta/\Delta x)^{d_{\text{old}}}}
$$

```python
# BinaryEmission._get_diff — majority rule + diff counts
for x in mut_set:
    sl = branch.lower_node.get_state(x)
    su = branch.upper_node.get_state(x)
    s0 = node.get_state(x)
    sm = 1 if (sl + su + s0 > 1.5) else 0    # majority rule
    d[0] += abs(sm - sl)   # lower sub-branch
    d[1] += abs(sm - su)   # upper sub-branch
    d[2] += abs(sm - s0)   # query sub-branch
    d[3] += abs(sl - su)   # old branch (without threading)
```

The BSP uses `PolarEmission` which extends this with an ancestral/derived penalty and a root reward factor.

### Traceback

After the forward pass, the BSP samples a path of joining branches by stochastic traceback. At each step it computes a "shrinkage" factor — the probability of not recombining — and uses it to decide how far back to walk before jumping to a new branch:

```python
# BSP._trace_back_helper
p = self._random()
q = 1.0
while x > interval.start_pos:
    rp = self._get_recomb_prob(self.rhos[x - 1], interval.time)
    non_recomb = (1.0 - rp) * self.forward_probs[x - 1][self.sample_index]
    all_prob = non_recomb + recomb_sum * interval.weight * rp / weight_sum
    shrinkage = non_recomb / all_prob
    q *= shrinkage
    if p >= q:
        return x    # recombination happened here → jump
    x -= 1
return interval.start_pos
```

## 3. The Time Sequence Propagator (TSP)

Conditioned on the BSP's sampled joining branches, the TSP is a forward HMM that answers: **at what time** should the new lineage coalesce on each branch?

### State space

For a single branch $b = [\ell, u]$, the TSP discretises the time axis into quantile-spaced intervals using the $\text{Exp}(1)$ CDF:

$$
q_k = F_{\text{Exp}}^{-1}\!\bigl(p_l + k \cdot \Delta p\bigr), \quad k = 0, \ldots, K
$$

where $p_l = 1 - e^{-\ell}$, $p_u = 1 - e^{-u}$, and $\Delta p = (p_u - p_l)/K$ with $K = \lceil (p_u - p_l) / q \rceil$ for gap parameter $q$ (default 0.02). This concentrates grid points where the coalescent density is highest (near the present).

```python
# pysinger/hmm/tsp.py — TSP._generate_grid
def _generate_grid(self, lb, ub):
    lq = 1.0 - math.exp(-lb)               # F_Exp(lb)
    uq = 1.0 - math.exp(-ub)               # F_Exp(ub)
    q = uq - lq
    n = math.ceil(q / self.gap)             # number of intervals
    points = [lb]
    for i in range(1, n):
        l = self._get_exp_quantile(lq + i * q / n)   # F_Exp^{-1}
        points.append(l)
    points.append(ub)
    return points

def _get_exp_quantile(self, p):
    return -math.log(1.0 - p)              # F_Exp^{-1}(p)
```

Each interval gets a representative time via the **exponential median** — the time where the $\text{Exp}(1)$ CDF is midway between the interval bounds:

$$
t^* = -\log\!\Bigl(1 - \tfrac{1}{2}\bigl[(1-e^{-l}) + (1-e^{-u})\bigr]\Bigr)
$$

```python
# pysinger/data/interval.py — Interval.fill_time
lq = 1.0 - math.exp(-lb)
uq = 1.0 - math.exp(-ub)
q = 0.5 * (lq + uq)           # midpoint in CDF space
self.time = -math.log(1.0 - q) # invert to get time
```

### Transition (PSMC kernel)

The TSP uses a **PSMC-style** transition kernel. Given current time $s$ and recombination rate $\rho$:

$$
P_{\text{PSMC}}(\rho, s, [t_1, t_2]) = \underbrace{\mathbb{1}[t_1 \leq s \leq t_2] \cdot e^{-\rho \ell}}_{\text{no recombination}} + \underbrace{\int_{t_1}^{t_2} f_{\text{PSMC}}(\rho, s, t)\,dt}_{\text{recombination + recoalescence}}
$$

The first term is a point mass (stay at the same time if no recombination occurs). The integral term accounts for recombination followed by re-coalescence at a new time $t$.

```python
# TSP._psmc_prob
def _psmc_prob(self, rho, s, t1, t2):
    l = 2.0 * s - self.lower_bound - self.cut_time

    # point mass: no recombination
    if t1 <= s <= t2:
        base = math.exp(-rho * l)
    else:
        base = 0.0

    # integral: recombination + recoalescence in [t1, t2]
    if t2 > t1:
        gap = max(self._psmc_cdf(rho, s, t2) - self._psmc_cdf(rho, s, t1), 0.0)
    else:
        gap = 0.0

    return max(0.0, min(1.0, base + gap))
```

The PSMC CDF itself integrates a piecewise density that depends on whether the target time is before or after the source time $s$:

```python
# TSP._psmc_cdf
def _psmc_cdf(self, rho, s, t):
    l = 2.0 * s - self.lower_bound - self.cut_time
    pre_factor = (1.0 - math.exp(-rho * l)) / l if l != 0 else rho

    if t <= s:
        integral = (2*t + math.exp(-t) * (math.exp(self.cut_time)
                    + math.exp(self.lower_bound))
                    - self.cut_time - self.lower_bound - 2.0)
    else:
        integral = (2*s + math.exp(self.cut_time - t)
                    + math.exp(self.lower_bound - t)
                    - 2*math.exp(s - t)
                    - self.cut_time - self.lower_bound)
    return pre_factor * integral
```

The transition matrix is **tridiagonal** (adjacent intervals interact most strongly), allowing efficient $O(K)$ forward updates. The diagonal, lower-diagonal, and upper-diagonal entries are normalised PSMC probabilities:

```python
# TSP._compute_diagonals — stay-in-place probability
for i, iv in enumerate(self.curr_intervals):
    base = self._psmc_prob(rho, iv.time, lb, ub)       # total mass
    diag = self._psmc_prob(rho, iv.time, iv.lb, iv.ub)  # self-mass
    self.diagonals[i] = diag / base if base > 0 else 0.0
```

The forward recursion uses `lower_sums` (cumulative from below) and `upper_sums` (cumulative from above) to avoid the $O(K^2)$ full matrix multiply:

$$
\alpha_i^{(x+1)} = L_i + D_i \cdot \alpha_i^{(x)} + U_i \cdot \sum_{j>i} \alpha_j^{(x)}
$$

```python
# TSP.forward
new_fp = list(self.lower_sums)
for i in range(self.dim):
    new_fp[i] += (
        self.diagonals[i] * self.forward_probs[self.curr_index - 1][i]
        + self.lower_diagonals[i] * self.upper_sums[i]
    )
    if self.curr_intervals[i].lb != self.curr_intervals[i].ub:
        new_fp[i] = max(1e-20, new_fp[i])    # numerical floor
self.forward_probs.append(new_fp)
```

### Transfer at topology changes

When the BSP switches branches at a recombination, the TSP handles three cases:

**Source → merging**: Collapse to a point mass at the deleted node's time, with regular intervals above and below.

```python
# TSP.transfer — source to merging case
t = r.deleted_node.time
t = max(lb_b, min(ub_b, t))                      # clamp to branch
self._generate_intervals(next_branch, lb_b, t)    # intervals below
self._generate_intervals(next_branch, t, t)        # point mass
if len(self.curr_intervals) > n_before:
    self._temp[-1] = 1.0                           # all mass on point
    self.curr_intervals[-1].node = r.deleted_node
self._generate_intervals(next_branch, t, ub_b)    # intervals above
```

**Target → recombined**: Expand from a point mass to the recombined branch's time range.

**Regular transfer**: Overlap intervals between old and new branches, preserving probability mass proportionally using exponential measure:

```python
# TSP._get_prop — proportion in exponential measure
def _get_prop(self, lb1, ub1, lb2, ub2):
    p1 = math.exp(-lb1) - math.exp(-ub1)   # exp measure of [lb1, ub1]
    p2 = math.exp(-lb2) - math.exp(-ub2)   # exp measure of [lb2, ub2]
    return p1 / p2 if p2 > 0 else 1.0
```

### Traceback

`sample_joining_nodes()` walks backward through the forward probabilities. At each step it samples the interval the coalescence came from and converts it to a `Node` at a jittered time:

```python
# TSP._sample_joining_node
def _sample_joining_node(self, interval):
    if interval.node is not None:
        return interval.node          # existing node (point mass)
    t = self._exp_median(interval.lb, interval.ub)   # jittered sample
    n = Node(time=t)
    n.index = _counter                # unique ID
    return n
```

The jitter (`_exp_median`) samples uniformly in the $[0.45, 0.55]$ quantile range of the exponential distribution, preventing all sampled times from clustering at the median:

```python
# TSP._exp_median
lq = 1.0 - math.exp(-lb)
uq = 1.0 - math.exp(-ub)
mq = (0.45 + 0.1 * self._random()) * (uq - lq) + lq   # jittered quantile
m = -math.log(1.0 - mq)
return max(lb, min(ub, m))
```

## 4. The MCMC

### Initialisation (`iterative_start`)

The initial ARG is built by threading haplotypes one by one:

1. Start with a singleton ARG (one sample connected to the root sentinel).
2. For each additional sample: run BSP → sample joining branches → run TSP → sample coalescence times → insert into ARG.
3. After all samples are threaded, rescale branch lengths.

```python
# pysinger/sampler.py — Sampler.iterative_start
def iterative_start(self, max_retries=5):
    for attempt in range(max_retries):
        try:
            self._build_singleton_arg()
            for node in self.ordered_sample_nodes[1:]:
                threader = self._make_threader()
                threader.thread(self.arg, node)
            self._rescale()
            return
        except RuntimeError:
            # HMM underflow — retry with fresh RNG
            self._rng = np.random.default_rng(self._seed + attempt + 1)
```

### Metropolis--Hastings moves (`internal_sample`)

Each MCMC iteration:

1. **Sample a cut point** $(x_0, b, t_c)$: pick a random position, branch, and time in the current ARG.
2. **Remove** the lineage passing through $(b, t_c)$ by tracing it forward and backward through recombination records.
3. **Propose** a new threading by running BSP + TSP on the modified ARG.
4. **Accept/reject** with Metropolis ratio:

$$
\alpha = \frac{h_{\text{old}}}{h_{\text{new}}}
$$

where $h$ is the effective tree height at the cut position.

```python
# pysinger/mcmc/threader.py — Threader.internal_rethread
def internal_rethread(self, arg, cut_point):
    self.cut_time = cut_point[2]
    arg.remove(cut_point)                    # extract lineage
    self._run_bsp(arg)                       # propose branch path
    self._sample_joining_branches(arg)
    self._run_tsp(arg)                       # propose coalescence times
    self._sample_joining_points(arg)
    ar = self._acceptance_ratio(arg)
    if self._random() < ar:
        arg.add(self.new_joining_branches, self.added_branches)   # accept
    else:
        arg.add(arg.joining_branches, arg.removed_branches)       # reject
    arg.approx_sample_recombinations()
    arg.clear_remove_info()
```

The acceptance ratio favours proposals that produce comparable or shorter tree heights:

```python
# Threader._acceptance_ratio
cut_height = max(child.time for child in arg.cut_tree.parents.keys())
old_height = cut_height
new_height = cut_height
# adjust for root-reaching branches...
if new_height <= 0:
    return 1.0
return old_height / new_height
```

### Rescaling

After each MCMC iteration, a global rescaling adjusts all internal node times so that the expected number of mutations matches the observed count:

$$
s = \frac{S_{\text{obs}}}{\mu_{\text{scaled}} \cdot L_{\text{total}}}
$$

where $S_{\text{obs}}$ is the number of segregating sites, $\mu_{\text{scaled}} = \mu \cdot N_e$, and $L_{\text{total}} = \sum_{\text{trees}} (\text{branch length} \times \text{genomic span})$.

```python
# pysinger/sampler.py — Sampler._rescale
total_obs = len({pos for n in self.sample_nodes
                 for pos in n.mutation_sites.keys() if pos >= 0})
total_branch = self.arg.get_arg_length()
expected = self.mut_rate * total_branch       # mu * Ne * L_total
scale = total_obs / expected

for n in internal_nodes:
    n.time *= scale
for pos, r in self.arg.recombinations.items():
    if 0 < pos < self.arg.sequence_length and r.start_time > 0:
        r.start_time *= scale
```

## 5. Ancestral state reconstruction (Fitch parsimony)

After threading, mutations need to be placed on branches. pysinger uses **Fitch parsimony** — a two-pass algorithm that minimises the number of mutation events.

**Up-pass (pruning)**: Bottom-up. For each internal node, merge children's states:

$$
s_p = \begin{cases}
s_{c_1} & \text{if } s_{c_1} = s_{c_2} \\
s_{c_i} & \text{if the other child is ambiguous} \\
0.5 & \text{if } s_{c_1} \neq s_{c_2} \text{ (both definite)}
\end{cases}
$$

```python
# pysinger/reconstruction/fitch.py — FitchReconstruction._fitch_up
def _fitch_up(self, c1, c2, p):
    s1 = self.pruning_node_states[c1]
    s2 = self.pruning_node_states[c2]
    if s1 == 0.5:
        s = s2              # c1 ambiguous → take c2
    elif s2 == 0.5:
        s = s1              # c2 ambiguous → take c1
    elif s1 != s2:
        s = 0.5             # disagree → ambiguous
    else:
        s = s1              # agree → take shared state
    self.pruning_node_states[p] = s
```

**Down-pass (peeling)**: Top-down. Resolve ambiguities by propagating the parent's definite state downward. The root sentinel resolves ambiguity to the ancestral state (0):

```python
# FitchReconstruction._fitch_down
def _fitch_down(self, parent, child):
    if parent.index == -1:
        # root sentinel: resolve to ancestral
        top_state = self.pruning_node_states[child]
        s = 0.0 if top_state == 0.5 else top_state
    else:
        sp = self.peeling_node_states[parent]
        sc = self.pruning_node_states[child]
        s = sp if sc == 0.5 else sc   # ambiguous → take parent
    self.peeling_node_states[child] = s
```

## 6. ARG encoding

The ARG is stored as a **sorted map** of `Recombination` records keyed by genomic position. Each record stores the set of branches deleted (existing before the breakpoint) and inserted (existing after).

The two main MCMC operations modify this map:

**`ARG.remove(cut_point)`**: Traces the cut lineage forward and backward through recombination records, modifying each record to excise the cut branch and its coalescence node. The key helper is `Recombination.trace_forward(t, branch)` which maps a branch across a topology change:

```python
# pysinger/data/recombination.py — Recombination.trace_forward
def trace_forward(self, t, curr_branch):
    if not self.affect(curr_branch):
        return curr_branch              # branch unchanged by this event
    if curr_branch == self.source_branch:
        if t >= self.start_time:
            return Branch()             # above recombination → lineage ends
        else:
            return self.recombined_branch
    elif curr_branch == self.target_branch:
        if t > self.inserted_node.time:
            return self.upper_transfer_branch
        else:
            return self.lower_transfer_branch
    else:
        return self.merging_branch
```

**`ARG.add(joining_branches, added_branches)`**: Threads a lineage back in by walking through the added positions and either modifying existing recombination records or creating new ones. After insertion, `_impute()` assigns allele states to the newly created coalescence nodes using majority rule.
