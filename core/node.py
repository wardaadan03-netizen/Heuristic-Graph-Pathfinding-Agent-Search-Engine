"""Node representation used by all search algorithms."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """A node in the grid graph."""
    row: int
    col: int
    g: float = float("inf")           # cost from start
    h: float = 0.0                    # heuristic to goal
    parent: Optional["Node"] = field(default=None, repr=False)

    @property
    def f(self) -> float:
        """Total estimated cost (A* key)."""
        return self.g + self.h

    @property
    def position(self) -> tuple[int, int]:
        return (self.row, self.col)

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.position == other.position

    def __lt__(self, other: "Node") -> bool:
        """Tie-break on f then h for consistent heap ordering."""
        return (self.f, self.h) < (other.f, other.h)