"""Dijkstra's algorithm — uniform-cost search without heuristic."""

from __future__ import annotations
import heapq
from typing import Optional

from algorithms.base import SearchAgent
from core.grid import Grid
from core.node import Node


class DijkstraAgent(SearchAgent):
    """
    Dijkstra's shortest-path algorithm.

    Equivalent to A* with a zero heuristic; expands nodes strictly
    by g(n).
    """

    name = "Dijkstra"

    def __init__(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]):
        super().__init__(grid, start, goal)

    def solve(self) -> Optional[list[tuple[int, int]]]:
        start_node = Node(*self.start, g=0.0, h=0.0)
        open_heap: list[Node] = [start_node]
        best_g: dict[tuple[int, int], float] = {self.start: 0.0}
        closed: set[tuple[int, int]] = set()

        while open_heap:
            current = heapq.heappop(open_heap)

            if current.position in closed:
                continue
            closed.add(current.position)
            self.expanded_nodes += 1

            if current.position == self.goal:
                self.path = self.reconstruct_path(current)
                self.path_cost = current.g
                self.found = True
                return self.path

            for neighbor in self.grid.neighbors(current):
                if neighbor.position in closed:
                    continue
                tentative_g = current.g + self.grid.cost(current, neighbor)
                if tentative_g < best_g.get(neighbor.position, float("inf")):
                    best_g[neighbor.position] = tentative_g
                    neighbor.g = tentative_g
                    neighbor.parent = current
                    heapq.heappush(open_heap, neighbor)

        self.found = False
        return None