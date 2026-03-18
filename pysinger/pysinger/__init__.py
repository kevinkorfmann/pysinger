"""
pysinger — pure-Python replica of the SINGER Bayesian ARG sampler.

Package layout mirrors the dependency order of the C++ codebase:

  data  →  rates  →  hmm  →  reconstruction  →  mcmc  →  sampler
"""
from .sampler import Sampler
from .parallel import ParallelSampler, run_parallel_chains

__all__ = ["Sampler", "ParallelSampler", "run_parallel_chains"]
