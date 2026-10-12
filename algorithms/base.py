"""Abstract base class shared by all search agents."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from core.grid import Grid


class SearchAgent(ABC):
    """Common interface for pathfinding agents."""

    name: str = "BaseAgent"

    def __init__(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]):
        if not grid.is_free(start):
            raise ValueError(f"Start {start} is blocked.")
        if not grid.is_free(goal):
            raise ValueError(f"Goal {goal} is blocked.")
        self.grid = grid
        self.start = start
        self.goal = goal

        # Metrics populated by subclasses
        self.expanded_nodes: int = 0
        self.path: Optional[list[tuple[int, int]]] = None
        self.path_cost: float = float("inf")
        self.found: bool = False

    @abstractmethod
    def solve(self) -> Optional[list[tuple[int, int]]]:
        """Return the path as a list of (row, col) tuples, or None."""
        raise NotImplementedError

    @staticmethod
    def reconstruct_path(node) -> list[tuple[int, int]]:
        """Walk parent pointers from goal back to start."""
        path = []
        cur = node
        while cur is not None:
            path.append(cur.position)
            cur = cur.parent
        return path[::-1]