"""
RateMap — piecewise-constant rate map (recombination or mutation).

Mirrors Rate_map.cpp / Rate_map.hpp.
"""
from __future__ import annotations

import bisect
from typing import List


class RateMap:
    """Piecewise-constant genetic rate map.

    Loaded from a file with lines:  left  right  rate

    Provides cumulative distance lookup for converting genomic positions
    to rate-weighted distances.
    """

    def __init__(self) -> None:
        self.coordinates: List[float] = []
        self.rate_distances: List[float] = [0.0]
        self.sequence_length: float = 0.0

    def load_map(self, filename: str) -> None:
        """Load a rate map from a whitespace-delimited file."""
        with open(filename) as fh:
            right = 0.0
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                left, right, rate = float(parts[0]), float(parts[1]), float(parts[2])
                self.coordinates.append(left)
                mut_dist = self.rate_distances[-1] + rate * (right - left)
                self.rate_distances.append(mut_dist)
        self.sequence_length = right
        self.coordinates.append(self.sequence_length)

    def _find_index(self, x: float) -> int:
        idx = bisect.bisect_right(self.coordinates, x) - 1
        return max(0, min(idx, len(self.coordinates) - 2))

    def cumulative_distance(self, x: float) -> float:
        idx = self._find_index(x)
        prev_dist = self.rate_distances[idx]
        next_dist = self.rate_distances[idx + 1]
        p = (x - self.coordinates[idx]) / (self.coordinates[idx + 1] - self.coordinates[idx])
        return (1 - p) * prev_dist + p * next_dist

    def segment_distance(self, x: float, y: float) -> float:
        return self.cumulative_distance(y) - self.cumulative_distance(x)

    def mean_rate(self) -> float:
        return self.rate_distances[-1] / self.sequence_length
