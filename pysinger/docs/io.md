# I/O and reconstruction

## VCF reader

```{eval-rst}
.. automodule:: pysinger.io.vcf_reader
   :members:
```

Two functions:

- **`read_vcf_phased(path, start, end)`**: Parses a phased VCF (`0|1` format). Each diploid individual produces 2 `Node` objects (one per haplotype). Each node's `mutation_sites` is populated with positions where it carries the derived allele.

- **`read_vcf_haploid(path, start, end)`**: Treats each sample column as a single haplotype (for haploid data or pre-split files).

Both return `(List[Node], sequence_length)`. The sequence length is the `end` parameter (or the last variant position + 1 if `end = inf`).

## tskit writer

```{eval-rst}
.. automodule:: pysinger.io.tskit_writer
   :members:
```

**`arg_to_tskit(arg, Ne)`** converts a pysinger `ARG` to a `tskit.TreeSequence`:

1. **Discover nodes**: Walk all marginal trees by replaying recombinations. Collect every node that appears in a parent/child relationship (excluding the root sentinel at `index = -1`).

2. **Add nodes to tskit**: Each node becomes a tskit node with `time = node.time * Ne` (converting coalescent units to generations). Sample nodes get `NODE_IS_SAMPLE` flag.

3. **Emit edges**: Replay recombinations again. For each tree interval $[x_k, x_{k+1})$, emit a tskit edge `(left, right, parent, child)` for every parent-child pair where the parent is not the root sentinel and `parent.time > child.time`.

4. **Sort and build**: Call `tables.sort()` and `tables.tree_sequence()`.

The resulting tree sequence can be used with the full tskit API for computing diversity, TMRCA, drawing trees, etc.

## Rate maps

```{eval-rst}
.. autoclass:: pysinger.rates.rate_map.RateMap
   :members:
```

`RateMap` stores a piecewise-constant rate function (recombination or mutation) as parallel arrays of `(left, right, rate)` intervals. Supports:

- `load_map(path)`: Read a 3-column file (left, right, rate).
- `cumulative_distance(pos)`: Integrated rate from 0 to `pos`.
- `segment_distance(a, b)`: Integrated rate from `a` to `b`.

When provided to the `Sampler`, rate maps override the scalar `recomb_rate` / `mut_rate` for computing per-bin `rhos` and `thetas`.

## Fitch parsimony reconstruction

```{eval-rst}
.. autoclass:: pysinger.reconstruction.fitch.FitchReconstruction
   :members:
```

**Fitch parsimony** assigns ancestral states to internal nodes with the minimum number of mutations. It runs on each marginal tree for each segregating site.

### Algorithm

**Pass 1 — Pruning (bottom-up):**
- Leaf nodes: state = observed allele.
- Internal nodes: if both children agree, take that state. If they disagree, mark as ambiguous (0.5). If one child is ambiguous, take the other's state.

**Pass 2 — Peeling (top-down):**
- Root: resolve ambiguity to ancestral (0).
- Internal nodes: if ambiguous, take parent's resolved state. If definite, keep it.

After both passes, `node.write_state(pos, state)` is called for each internal node, updating the node's `mutation_sites` map. This is used during `ARG._impute()` to assign allele states to newly threaded coalescence nodes.
