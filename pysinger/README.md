# pysinger

A pure-Python rewrite of the [SINGER](https://github.com/popgenmethods/SINGER) Bayesian ARG sampler. It reimplements the full BSP + TSP threading and MCMC sampling pipeline in readable Python, and can export inferred ARGs directly to `tskit.TreeSequence` for downstream analysis.

> Deng, Y., Nielsen, R. & Song, Y.S. Robust and accurate Bayesian inference of genome-wide genealogies for hundreds of genomes. *Nature Genetics* **57**, 2124--2135 (2025). [https://doi.org/10.1038/s41588-025-02317-9](https://doi.org/10.1038/s41588-025-02317-9)

## Quick start

Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv sync --extra demo          # creates .venv and installs all deps
# or without uv:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[demo]"
```

```python
from pysinger import Sampler
from pysinger.io.tskit_writer import arg_to_tskit

sampler = Sampler(Ne=10000, recomb_rate=1e-8, mut_rate=1e-8)
sampler.set_seed(42)
sampler.load_vcf("data.vcf", start=0, end=1_000_000)
sampler.iterative_start()
sampler.internal_sample(num_iters=1000, spacing=1)
ts = arg_to_tskit(sampler.arg, Ne=10000)
```

See `demo.ipynb` for a full walkthrough using a stdpopsim zigzag simulation with convergence diagnostics and validation plots.

## Package structure

```
pysinger/
├── sampler.py           # Top-level MCMC sampler
├── data/                # ARG, Node, Branch, Tree, Recombination
├── hmm/                 # BSP (branch HMM), TSP (time HMM), emissions
├── mcmc/                # Threader (BSP + TSP threading)
├── io/                  # VCF reader, tskit writer
├── rates/               # Piecewise recombination/mutation rate maps
└── reconstruction/      # Fitch parsimony for ancestral states
```
