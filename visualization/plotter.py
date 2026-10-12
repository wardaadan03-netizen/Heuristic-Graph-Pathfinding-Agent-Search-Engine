"""Matplotlib-based grid and path visualizations."""

from __future__ import annotations
import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.grid import Grid


class PathPlotter:
    """Renders mazes, paths, and comparison dashboards."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ---------- Single path plot ----------
    def plot_path(
        self,
        grid: Grid,
        path: Optional[list[tuple[int, int]]],
        start: tuple[int, int],
        goal: tuple[int, int],
        title: str,
        filename: str,
        expanded_nodes: int = 0,
    ) -> str:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(grid.matrix, cmap="Greys", origin="upper", interpolation="nearest")

        if path and len(path) > 1:
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ax.plot(xs, ys, color="#1f77b4", linewidth=2.2, label="Path")
            ax.scatter(xs, ys, color="#1f77b4", s=8, zorder=3)

        ax.scatter(*start[::-1], color="#2ca02c", s=180, marker="o",
                   edgecolors="black", zorder=4, label="Start")
        ax.scatter(*goal[::-1], color="#d62728", s=180, marker="*",
                   edgecolors="black", zorder=4, label="Goal")

        ax.set_title(f"{title}\nExpanded nodes: {expanded_nodes}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=9)

        out_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------- Comparison dashboard ----------
    def plot_comparison(
        self,
        grids_paths: list[tuple[str, Grid, Optional[list[tuple[int, int]]], int]],
        start: tuple[int, int],
        goal: tuple[int, int],
        filename: str = "comparison.png",
    ) -> str:
        n = len(grids_paths)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows))
        axes = np.atleast_1d(axes).flatten()

        for ax, (name, grid, path, expanded) in zip(axes, grids_paths):
            ax.imshow(grid.matrix, cmap="Greys", origin="upper", interpolation="nearest")
            if path and len(path) > 1:
                ys = [p[0] for p in path]; xs = [p[1] for p in path]
                ax.plot(xs, ys, color="#1f77b4", linewidth=2.0)
            ax.scatter(*start[::-1], color="#2ca02c", s=120, marker="o",
                       edgecolors="black", zorder=4)
            ax.scatter(*goal[::-1], color="#d62728", s=140, marker="*",
                       edgecolors="black", zorder=4)
            ax.set_title(f"{name}\nExpanded: {expanded}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

        for ax in axes[n:]:
            ax.axis("off")

        out_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------- Bar chart comparison ----------
    def plot_metrics_bar(
        self,
        summary: dict,
        filename: str = "metrics_bar.png",
    ) -> str:
        algos = list(summary.keys())
        runtime = [summary[a]["avg_runtime_ms"] for a in algos]
        expanded = [summary[a]["avg_expanded_nodes"] for a in algos]
        path_len = [summary[a]["avg_path_length"] for a in algos]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(algos)]

        axes[0].bar(algos, runtime, color=colors)
        axes[0].set_title("Avg Runtime (ms)")
        axes[1].bar(algos, expanded, color=colors)
        axes[1].set_title("Avg Expanded Nodes")
        axes[2].bar(algos, path_len, color=colors)
        axes[2].set_title("Avg Path Length")

        for ax in axes:
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        out_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path