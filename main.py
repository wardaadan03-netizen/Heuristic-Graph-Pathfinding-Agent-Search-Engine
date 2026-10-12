"""
Entry point for the Heuristic Graph Pathfinding Agent Search Engine.

Usage:
    python main.py                # run full benchmark suite
    python main.py --demo         # single demo grid with all algorithms
    python main.py --fast         # skip Q-Learning (fast mode)
"""

from __future__ import annotations
import argparse
import os

from benchmarks import BenchmarkRunner
from algorithms import AStarAgent, DijkstraAgent, QLearningAgent
from core.grid import Grid
from core.heuristics import manhattan
from metrics import MetricsTracker
from visualization import PathPlotter


def run_demo(output_root: str = "outputs") -> None:
    """Single-grid demo showing A*, Dijkstra and Q-Learning side-by-side."""
    print("[Demo] Building 25x25 grid with 25% obstacles...")
    grid = Grid.random(25, 25, obstacle_density=0.25, seed=7)
    start, goal = (0, 0), (24, 24)
    grid.matrix[start] = 0
    grid.matrix[goal] = 0

    tracker = MetricsTracker()
    plotter = PathPlotter(os.path.join(output_root, "figures"))

    astar = AStarAgent(grid, start, goal, heuristic=manhattan)
    tracker.run_agent(astar, obstacle_density=0.25, seed=7, heuristic="manhattan")

    dijkstra = DijkstraAgent(grid, start, goal)
    tracker.run_agent(dijkstra, obstacle_density=0.25, seed=7, heuristic="zero")

    ql = QLearningAgent(grid, start, goal, seed=7, episodes=600)
    tracker.run_agent(ql, obstacle_density=0.25, seed=7)

    plotter.plot_comparison(
        [
            ("A*", grid, astar.path, astar.expanded_nodes),
            ("Dijkstra", grid, dijkstra.path, dijkstra.expanded_nodes),
            ("Q-Learning", grid, ql.path, ql.expanded_nodes),
        ],
        start, goal,
        filename="demo_comparison.png",
    )

    tracker.export(os.path.join(output_root, "logs"))
    print(f"[Demo] Visualisation saved -> {output_root}/figures/demo_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heuristic Graph Pathfinding Agent Search Engine"
    )
    parser.add_argument("--demo", action="store_true", help="Run a single demo grid.")
    parser.add_argument("--fast", action="store_true", help="Skip Q-Learning.")
    parser.add_argument("--output", default="outputs", help="Output root directory.")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.output)
        return

    runner = BenchmarkRunner(output_root=args.output)
    runner.run_all(include_qlearning=not args.fast)


if __name__ == "__main__":
    main()