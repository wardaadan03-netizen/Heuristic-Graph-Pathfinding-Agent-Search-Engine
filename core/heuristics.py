"""Heuristic functions for informed search."""

from __future__ import annotations
import math


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """L1 distance — admissible for 4-connected grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """L2 distance — admissible but weaker for 4-connected grids."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> float:
    """L∞ distance — admissible for 8-connected grids."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def zero(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Null heuristic — reduces A* to Dijkstra."""
    return 0.0


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "zero": zero,
}