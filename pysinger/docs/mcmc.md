# MCMC threading

The MCMC module (`pysinger.mcmc`) contains the `Threader` class that combines the BSP and TSP into a single threading operation, used both for initial ARG construction and for MCMC re-threading moves.

## Threader

```{eval-rst}
.. autoclass:: pysinger.mcmc.threader.Threader
   :members:
```

### `thread(arg, node)` — initial threading

Used during `iterative_start()` to add a new leaf node to the ARG:

```
arg.add_sample(node)         # register node, set up removed_branches
├─ BSP forward pass          # which branch to join at each position?
├─ sample_joining_branches   # stochastic traceback → pos → Branch map
├─ TSP forward pass          # when to coalesce on each branch?
├─ sample_joining_points     # stochastic traceback → pos → Node map
├─ arg.add(joining, added)   # thread the lineage into the ARG
├─ arg.approx_sample_recombs # assign recombination times
└─ arg.clear_remove_info     # clean up working state
```

### `internal_rethread(arg, cut_point)` — MCMC move

Used during `internal_sample()` for Metropolis--Hastings re-threading:

```
arg.remove(cut_point)        # extract lineage → removed_branches, joining_branches
├─ BSP forward pass          # propose new branch path
├─ sample_joining_branches
├─ TSP forward pass          # propose new coalescence times
├─ sample_joining_points
├─ acceptance_ratio          # Metropolis ratio = h_old / h_new
├─ if accept:
│    arg.add(new_joining, new_added)
│  else:
│    arg.add(old_joining, old_removed)     # restore original
├─ arg.approx_sample_recombs
└─ arg.clear_remove_info
```

### Acceptance ratio

The Metropolis ratio is:

$$
\alpha = \frac{h_{\text{old}}}{h_{\text{new}}}
$$

where $h$ is the **effective tree height** at the cut position. For the cut tree, this is the maximum child node time. If the joining branch reaches the root sentinel, $h$ is replaced by the coalescence node time. This favours proposals that produce comparable or shorter tree heights.

### Emission model assignment

The BSP uses `PolarEmission` (accounts for ancestral/derived polarisation), while the TSP uses `BinaryEmission` (symmetric model). This matches the C++ implementation where the branch-level HMM benefits from polarity information but the time-level HMM does not.

## Sampler

```{eval-rst}
.. autoclass:: pysinger.sampler.Sampler
   :members:
```

### `iterative_start()`

1. Build a singleton ARG with the first sample node.
2. For each remaining sample, create a `Threader` and call `thread(arg, node)`.
3. Rescale all internal node times so that expected mutations match observed.

Includes retry logic: if a threading step fails due to HMM underflow (more likely on long sequences), the entire build is retried with a fresh RNG state.

### `internal_sample(num_iters, spacing)`

For each iteration:
1. Repeat until at least `spacing * sequence_length` bp have been updated:
   - Sample a random cut point.
   - Create a `Threader` and call `internal_rethread(arg, cut_point)`.
   - If the rethread fails (RuntimeError from underflow), restore the original ARG state and break.
2. Compute `last_arg_length` (pre-rescale).
3. Rescale all node times and recombination start times.

### `_rescale()`

Computes a global scale factor:

$$
s = \frac{S_{\text{obs}}}{\theta_{\text{scaled}} \cdot L_{\text{total}}}
$$

Multiplies all internal node times and recombination `start_time` values by $s$. This keeps coalescence times calibrated to the mutation rate throughout the MCMC.
