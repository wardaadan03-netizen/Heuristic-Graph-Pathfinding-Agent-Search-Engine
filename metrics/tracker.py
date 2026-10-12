"""Collect, aggregate, and export algorithm run metrics."""

from __future__ import annotations
import csv
import json
import os
import time
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from algorithms.base import SearchAgent


@dataclass
class RunResult:
    """Single (algorithm, scenario) run result."""
    algorithm: str
    grid_rows: int
    grid_cols: int
    obstacle_density: float
    seed: int
    start: tuple[int, int]
    goal: tuple[int, int]
    found: bool
    path_length: int
    path_cost: float
    expanded_nodes: int
    total_free_cells: int
    expanded_density: float       # expanded / free cells
    runtime_ms: float
    heuristic: Optional[str] = None
    notes: str = ""


class MetricsTracker:
    """Collects RunResult rows and exports logs to CSV/JSON."""

    def __init__(self):
        self.results: list[RunResult] = []

    # ---------- Execution wrapper ----------
    def run_agent(
        self,
        agent: SearchAgent,
        obstacle_density: float,
        seed: int,
        heuristic: Optional[str] = None,
        notes: str = "",
    ) -> RunResult:
        total_free = int((agent.grid.matrix == 0).sum())

        t0 = time.perf_counter()
        path = agent.solve()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        path_len = len(path) - 1 if path else 0
        density = agent.expanded_nodes / total_free if total_free else 0.0

        result = RunResult(
            algorithm=agent.name,
            grid_rows=agent.grid.rows,
            grid_cols=agent.grid.cols,
            obstacle_density=obstacle_density,
            seed=seed,
            start=agent.start,
            goal=agent.goal,
            found=agent.found,
            path_length=path_len,
            path_cost=agent.path_cost if agent.found else float("inf"),
            expanded_nodes=agent.expanded_nodes,
            total_free_cells=total_free,
            expanded_density=density,
            runtime_ms=elapsed_ms,
            heuristic=heuristic,
            notes=notes,
        )
        self.results.append(result)
        return result

    # ---------- Aggregation ----------
    def summary(self) -> dict:
        """Aggregate metrics per algorithm."""
        agg: dict[str, dict] = {}
        for r in self.results:
            bucket = agg.setdefault(
                r.algorithm,
                {
                    "runs": 0,
                    "successes": 0,
                    "avg_runtime_ms": 0.0,
                    "avg_expanded_nodes": 0.0,
                    "avg_path_length": 0.0,
                    "avg_expanded_density": 0.0,
                },
            )
            bucket["runs"] += 1
            bucket["successes"] += int(r.found)
            bucket["avg_runtime_ms"] += r.runtime_ms
            bucket["avg_expanded_nodes"] += r.expanded_nodes
            bucket["avg_path_length"] += r.path_length
            bucket["avg_expanded_density"] += r.expanded_density

        for algo, bucket in agg.items():
            n = bucket["runs"] or 1
            bucket["avg_runtime_ms"] /= n
            bucket["avg_expanded_nodes"] /= n
            bucket["avg_path_length"] /= n
            bucket["avg_expanded_density"] /= n
            bucket["success_rate"] = bucket["successes"] / n
        return agg

       # ---------- Export ----------
    def export(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)

        # ---------- CSV (unchanged) ----------
        csv_path = os.path.join(log_dir, "run_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "algorithm", "grid_rows", "grid_cols", "obstacle_density",
                "seed", "start", "goal", "found", "path_length", "path_cost",
                "expanded_nodes", "total_free_cells", "expanded_density",
                "runtime_ms", "heuristic", "notes",
            ])
            for r in self.results:
                # FIX: CSV — write "null" instead of "inf" for readability
                cost = r.path_cost if math.isfinite(r.path_cost) else "null"
                writer.writerow([
                    r.algorithm, r.grid_rows, r.grid_cols, r.obstacle_density,
                    r.seed, r.start, r.goal, r.found, r.path_length, cost,
                    r.expanded_nodes, r.total_free_cells, r.expanded_density,
                    round(r.runtime_ms, 4), r.heuristic or "", r.notes,
                ])

        # ---------- JSON (FIXED) ----------
        json_path = os.path.join(log_dir, "run_metrics.json")
        with open(json_path, "w") as f:
            json.dump(
                [self._row_to_json(r) for r in self.results],
                f,
                indent=2,
                allow_nan=False,   # strict JSON: no Infinity / NaN tokens
            )

        # ---------- Summary JSON ----------
        summary_path = os.path.join(log_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                self.summary(),
                f,
                indent=2,
                allow_nan=False,
                default=self._sanitize,
            )

        print(f"[MetricsTracker] Exported CSV  -> {csv_path}")
        print(f"[MetricsTracker] Exported JSON -> {json_path}")
        print(f"[MetricsTracker] Exported summary -> {summary_path}")

    # ---------- JSON helpers ----------
    @staticmethod
    def _row_to_json(r) -> dict:
        """Convert a RunResult to a JSON-safe dict (inf/nan -> null)."""
        d = asdict(r)
        for key, val in d.items():
            if isinstance(val, float) and not math.isfinite(val):
                d[key] = None          # "Infinity" -> null
            elif isinstance(val, tuple):
                d[key] = list(val)     # tuples -> JSON arrays
        return d

    @staticmethod
    def _sanitize(obj):
        """Fallback for objects json can't serialize natively."""
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return str(obj)