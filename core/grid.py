"""Grid environment with maze generation and neighbor expansion."""

from __future__ import annotations
import random
from typing import Iterator
import numpy as np

from core.node import Node


class Grid:
    """
    A 2D grid where 0 = free cell and 1 = obstacle.

    Supports 4-directional movement.
    """

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

    def __init__(self, matrix: np.ndarray):
        if matrix.ndim != 2:
            raise ValueError("Grid matrix must be 2D.")
        self.matrix = matrix
        self.rows, self.cols = matrix.shape

    # ---------- Constructors ----------
    @classmethod
    def random(
        cls,
        rows: int,
        cols: int,
        obstacle_density: float = 0.25,
        seed: int | None = None,
        start: tuple[int, int] | None = None,
        goal: tuple[int, int] | None = None,
    ) -> "Grid":
        """Generate a random grid that guarantees start/goal are free."""
        rng = random.Random(seed)
        matrix = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                if rng.random() < obstacle_density:
                    matrix[r, c] = 1

        if start:
            matrix[start] = 0
        if goal:
            matrix[goal] = 0

        return cls(matrix)

    # ---------- Queries ----------
    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, pos: tuple[int, int]) -> bool:
        return self.in_bounds(pos) and self.matrix[pos] == 0

    def neighbors(self, node: Node) -> Iterator[Node]:
        """Yield valid 4-connected neighbor nodes."""
        for dr, dc in self.DIRECTIONS:
            nr, nc = node.row + dr, node.col + dc
            if self.is_free((nr, nc)):
                yield Node(nr, nc)

    def cost(self, a: Node, b: Node) -> float:
        """Uniform cost for 4-connected grids."""
        return 1.0

    def __repr__(self) -> str:
        return f"Grid(rows={self.rows}, cols={self.cols})"