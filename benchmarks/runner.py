"""Benchmark suite: runs algorithms across varying grid obstacle arrays."""

from __future__ import annotations
import os
from typing import Optional
from tabulate import tabulate

from algorithms import AStarAgent, DijkstraAgent, QLearningAgent
from config.settings import CONFIG
from core.grid import Grid
from core.heuristics import manhattan
from metrics import MetricsTracker
from visualization import PathPlotter


SCENARIOS = [
    # (rows, cols, obstacle_density, seed)
    (20, 20, 0.10, 1),
    (20, 20, 0.25, 2),
    (30, 30, 0.20, 3),
    (30, 30, 0.30, 4),
    (40, 40, 0.25, 5),
]


class BenchmarkRunner:
    """Coordinates scenarios, agents, metrics, and visualization."""

    def __init__(self, output_root: str = "outputs"):
        self.output_root = output_root
        self.tracker = MetricsTracker()
        self.plotter = PathPlotter(os.path.join(output_root, "figures"))
        os.makedirs(os.path.join(output_root, "logs"), exist_ok=True)

    # ---------- Single scenario ----------
    def run_scenario(
        self,
        rows: int,
        cols: int,
        obstacle_density: float,
        seed: int,
        include_qlearning: bool = True,
        scenario_id: int = 0,
    ) -> list[tuple]:
        """Returns list of (name, grid, path, expanded) for comparison plot."""
        grid = Grid.random(rows, cols, obstacle_density, seed=seed)
        start = (0, 0)
        goal = (rows - 1, cols - 1)

        # Guarantee solvability is not required — some paths may legitimately fail.
        grid.matrix[start] = 0
        grid.matrix[goal] = 0

        results_for_plot = []

        # --- A* ---
        astar = AStarAgent(grid, start, goal, heuristic=manhattan)
        self.tracker.run_agent(
            astar,
            obstacle_density=obstacle_density,
            seed=seed,
            heuristic="manhattan",
            notes=f"scenario#{scenario_id}",
        )
        results_for_plot.append(("A*", grid, astar.path, astar.expanded_nodes))

        # --- Dijkstra ---
        dijkstra = DijkstraAgent(grid, start, goal)
        self.tracker.run_agent(
            dijkstra,
            obstacle_density=obstacle_density,
            seed=seed,
            heuristic="zero",
            notes=f"scenario#{scenario_id}",
        )
        results_for_plot.append(("Dijkstra", grid, dijkstra.path, dijkstra.expanded_nodes))

        # --- Q-Learning ---
        if include_qlearning:
            ql = QLearningAgent(grid, start, goal, seed=seed)
            self.tracker.run_agent(
                ql,
                obstacle_density=obstacle_density,
                seed=seed,
                heuristic=None,
                notes=f"scenario#{scenario_id} (episodes={ql.episodes})",
            )
            results_for_plot.append(("Q-Learning", grid, ql.path, ql.expanded_nodes))

        # Save comparison figure for this scenario
        self.plotter.plot_comparison(
            results_for_plot, start, goal,
            filename=f"scenario_{scenario_id}_{rows}x{cols}_d{int(obstacle_density*100)}.png",
        )
        return results_for_plot

    # ---------- Full benchmark ----------
    def run_all(self, include_qlearning: bool = True) -> None:
        for i, (rows, cols, density, seed) in enumerate(SCENARIOS):
            print(f"\n[Benchmark] Scenario {i}: {rows}x{cols} | density={density} | seed={seed}")
            self.run_scenario(rows, cols, density, seed,
                              include_qlearning=include_qlearning, scenario_id=i)

        # Export metrics
        self.tracker.export(os.path.join(self.output_root, "logs"))

        # Bar-chart comparison
        summary = self.tracker.summary()
        self.plotter.plot_metrics_bar(summary, filename="metrics_bar.png")

        # Pretty console table
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        table = []
        for algo, s in summary.items():
            table.append([
                algo,
                f"{s['success_rate']*100:.1f}%",
                f"{s['avg_runtime_ms']:.2f}",
                f"{s['avg_expanded_nodes']:.1f}",
                f"{s['avg_path_length']:.1f}",
                f"{s['avg_expanded_density']*100:.1f}%",
            ])
        print(tabulate(
            table,
            headers=["Algorithm", "Success", "Runtime (ms)", "Expanded",
                     "Path Len", "Expanded Density"],
            tablefmt="github",
        ))
        print(f"\nLogs saved to : {os.path.abspath(self.output_root + '/logs')}")
        print(f"Figures saved : {os.path.abspath(self.output_root + '/figures')}")