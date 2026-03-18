# Overview

## What pysinger does

pysinger performs **Bayesian posterior sampling of Ancestral Recombination Graphs (ARGs)** under the Sequentially Markov Coalescent (SMC). Given a phased VCF file and population-genetic parameters ($N_e$, $\mu$, $r$), it:

1. **Builds an initial ARG** by iteratively threading each haplotype through the growing graph using two coupled Hidden Markov Models.
2. **Refines the ARG** via Metropolis--Hastings MCMC that proposes local re-threading moves.
3. **Exports** the inferred ARG as a `tskit.TreeSequence` for downstream population-genetic analysis.

## The SINGER algorithm in one paragraph

SINGER threads one haplotype at a time into a partially-built ARG. For each new haplotype, a **Branch Sequence Propagator (BSP)** runs a forward HMM over the genome to decide *which branch* in each marginal tree the new lineage should join. Conditioned on that, a **Time Sequence Propagator (TSP)** runs a second forward HMM to decide *when* (at what coalescence time) it should join. Both HMMs use mutation data as emission evidence. After all haplotypes are threaded, MCMC iterates: pick a random lineage, remove it, and re-thread using BSP + TSP with Metropolis acceptance.

## Pipeline

```
VCF file
  │
  ▼
Sampler.load_vcf()        ← parse phased genotypes into Node objects
  │
  ▼
Sampler.iterative_start() ← thread haplotypes 1-by-1 (BSP + TSP)
  │
  ▼
Sampler.internal_sample() ← MCMC re-threading with Metropolis--Hastings
  │
  ▼
arg_to_tskit()            ← export to tskit.TreeSequence
```

## Package map

| Module | Role |
|--------|------|
| `pysinger.sampler` | Top-level orchestrator |
| `pysinger.data` | Core data structures: `Node`, `Branch`, `Tree`, `Recombination`, `ARG`, `Interval` |
| `pysinger.hmm` | Forward HMMs: `BSP` (branch), `TSP` (time), `CoalescentCalculator`, emission models |
| `pysinger.mcmc` | `Threader` — combines BSP + TSP into a single threading/re-threading operation |
| `pysinger.io` | VCF reader, tskit writer |
| `pysinger.rates` | Piecewise-constant recombination/mutation rate maps |
| `pysinger.reconstruction` | Fitch parsimony for ancestral state reconstruction |
