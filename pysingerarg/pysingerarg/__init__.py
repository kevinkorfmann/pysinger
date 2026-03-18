"""
pysingerarg — pure-Python replica of the SINGER Bayesian ARG sampler.

Package layout mirrors the dependency order of the C++ codebase:

  data  →  rates  →  hmm  →  reconstruction  →  mcmc  →  sampler
"""
from .sampler import Sampler

__all__ = ["Sampler"]
