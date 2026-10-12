"""A* search with pluggable heuristic — implemented from scratch."""

from __future__ import annotations
import heapq
from typing import Callable, Optional

from algorithms.base import SearchAgent
from core.grid import Grid
from core.heuristics import manhattan
from core.node import Node


class AStarAgent(SearchAgent):
    """
    A* Search.

    f(n) = g(n) + h(n)
    Uses a binary heap (heapq) as the open set and a dict-based closed set.
    """

    name = "A*"

    def __init__(
        self,
        grid: Grid,
        start: tuple[int, int],
        goal: tuple[int, int],
        heuristic: Callable[[tuple[int, int], tuple[int, int]], float] = manhattan,
    ):
        super().__init__(grid, start, goal)
        self.heuristic = heuristic

    def solve(self) -> Optional[list[tuple[int, int]]]:
        start_node = Node(*self.start, g=0.0, h=self.heuristic(self.start, self.goal))
        open_heap: list[Node] = [start_node]
        # best g-score per position
        g_score: dict[tuple[int, int], float] = {self.start: 0.0}
        closed: set[tuple[int, int]] = set()

        while open_heap:
            current = heapq.heappop(open_heap)

            if current.position in closed:
                continue
            closed.add(current.position)
            self.expanded_nodes += 1

            # Goal reached
            if current.position == self.goal:
                self.path = self.reconstruct_path(current)
                self.path_cost = current.g
                self.found = True
                return self.path

            for neighbor in self.grid.neighbors(current):
                if neighbor.position in closed:
                    continue
                tentative_g = current.g + self.grid.cost(current, neighbor)
                if tentative_g < g_score.get(neighbor.position, float("inf")):
                    g_score[neighbor.position] = tentative_g
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor.position, self.goal)
                    neighbor.parent = current
                    heapq.heappush(open_heap, neighbor)

        self.found = False
        return None