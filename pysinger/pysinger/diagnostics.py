"""
Convergence diagnostics for parallel MCMC chains.

Provides Gelman-Rubin R-hat and effective sample size (ESS) calculations
to assess convergence across multiple chains producing tskit TreeSequences.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def tree_heights(ts, num_points: int = 100) -> np.ndarray:
    """Extract TMRCA at evenly spaced genomic positions.

    Parameters
    ----------
    ts : tskit.TreeSequence
        A tree sequence from one chain.
    num_points : int
        Number of positions to sample.

    Returns
    -------
    heights : ndarray of shape (num_points,)
    """
    positions = np.linspace(0, ts.sequence_length, num_points, endpoint=False)
    heights = np.empty(num_points)
    for i, pos in enumerate(positions):
        tree = ts.at(pos)
        heights[i] = max(ts.node(u).time for u in tree.roots)
    return heights


def total_branch_lengths(ts, num_points: int = 100) -> np.ndarray:
    """Extract total branch length at evenly spaced positions."""
    positions = np.linspace(0, ts.sequence_length, num_points, endpoint=False)
    lengths = np.empty(num_points)
    for i, pos in enumerate(positions):
        tree = ts.at(pos)
        lengths[i] = tree.total_branch_length
    return lengths


def gelman_rubin(chains: np.ndarray) -> float:
    """Compute the Gelman-Rubin R-hat statistic.

    Parameters
    ----------
    chains : ndarray of shape (n_chains, n_samples)
        Each row is one chain's samples for a scalar summary.

    Returns
    -------
    r_hat : float
        Values close to 1.0 indicate convergence.  Typically R-hat < 1.05
        is considered acceptable.
    """
    n_chains, n_samples = chains.shape
    if n_chains < 2:
        return float("nan")

    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    grand_mean = chain_means.mean()

    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)
    # Within-chain variance
    W = np.mean(chain_vars)

    if W == 0:
        return float("nan")

    # Pooled variance estimate
    var_hat = (1 - 1 / n_samples) * W + (1 / n_samples) * B
    r_hat = np.sqrt(var_hat / W)
    return float(r_hat)


def effective_sample_size(chain: np.ndarray, max_lag: int = 100) -> float:
    """Estimate ESS from a single chain using autocorrelation.

    Parameters
    ----------
    chain : ndarray of shape (n_samples,)
    max_lag : int
        Maximum lag for autocorrelation estimation.

    Returns
    -------
    ess : float
    """
    n = len(chain)
    if n < 4:
        return float(n)

    mean = chain.mean()
    var = chain.var()
    if var == 0:
        return float(n)

    # Compute autocorrelation using FFT
    centered = chain - mean
    fft = np.fft.fft(centered, n=2 * n)
    acf = np.fft.ifft(fft * np.conj(fft)).real[:n] / (var * n)

    # Sum consecutive pairs of autocorrelations (Geyer's initial positive
    # sequence estimator); stop when a pair sums to negative.
    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        rho = acf[lag]
        if lag % 2 == 1:
            pair_sum = acf[lag - 1] + rho
            if pair_sum < 0:
                break
        tau += 2 * rho

    return n / tau


def convergence_summary(
    tree_sequences: list,
    num_points: int = 100,
) -> dict:
    """Compute convergence diagnostics across chains.

    Parameters
    ----------
    tree_sequences : list of tskit.TreeSequence
        One per chain.
    num_points : int
        Number of genomic positions to sample for summary statistics.

    Returns
    -------
    summary : dict
        Keys: ``r_hat_height``, ``r_hat_branch_length``, ``ess_per_chain``,
        ``mean_heights``, ``mean_branch_lengths``.
    """
    n_chains = len(tree_sequences)

    height_chains = np.array([tree_heights(ts, num_points) for ts in tree_sequences])
    branch_chains = np.array(
        [total_branch_lengths(ts, num_points) for ts in tree_sequences]
    )

    # R-hat on the mean statistic across positions
    r_hat_h = gelman_rubin(height_chains)
    r_hat_bl = gelman_rubin(branch_chains)

    # Per-chain ESS (using mean height as the trace)
    ess = [effective_sample_size(height_chains[i]) for i in range(n_chains)]

    return {
        "r_hat_height": r_hat_h,
        "r_hat_branch_length": r_hat_bl,
        "ess_per_chain": ess,
        "mean_heights": height_chains.mean(axis=1).tolist(),
        "mean_branch_lengths": branch_chains.mean(axis=1).tolist(),
    }
