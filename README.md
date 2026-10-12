# Heuristic Graph Pathfinding Agent Search Engine

A professional, from-scratch implementation of grid-based pathfinding agents
(A*, Dijkstra, Q-Learning) with benchmarking, metrics tracking, and visual logs.

## Features

- **From-scratch algorithms** — no external graph libraries (only numpy for arrays,
  matplotlib for plots). A* and Dijkstra use a binary heap + dict-based g-scores;
  Q-Learning implements tabular TD updates with ε-greedy exploration.
- **Pluggable heuristics** — Manhattan, Euclidean, Chebyshev, and zero (null).
- **Reproducible scenarios** — every benchmark is seeded.
- **Rich metrics** — runtime (ms), expanded node count, expanded density
  (expanded / free cells), path length, success rate.
- **Visualization** — per-scenario comparison dashboards, aggregate bar charts.
- **Export logs** — CSV, JSON, and summary JSON written to `outputs/logs/`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full benchmark suite (5 scenarios × 3 algorithms)
python main.py

# Fast mode (skip Q-Learning)
python main.py --fast

# Single demo grid with side-by-side comparison
python main.py --demo
```

## Output Structure

```
outputs/
├── logs/
│   ├── run_metrics.csv        # one row per run
│   ├── run_metrics.json       # full structured dump
│   └── summary.json           # per-algorithm averages
└── figures/
    ├── scenario_0_20x20_d10.png
    ├── ...
    ├── comparison.png
    └── metrics_bar.png
```

## Metrics Definitions

| Metric              | Definition                                              |
|---------------------|---------------------------------------------------------|
| Runtime (ms)        | Wall-clock time for `solve()` only (excl. grid setup).  |
| Expanded Nodes      | Pop operations from the open set (A*/Dijkstra) or steps taken by greedy rollout (QL). |
| Expanded Density    | Expanded nodes ÷ total free cells.                      |
| Path Length         | Number of edges in the returned path.                   |
| Success Rate        | Fraction of runs that reached the goal.                 |

## Algorithm Notes

# A* Search
- Open set: `heapq` on `(f, h)` tuples (tie-break favors deeper nodes).
- Closed set: Python `set` on positions.
- Admissible for 4-connected grids with Manhattan heuristic → optimal.

### Dijkstra
- A* with null heuristic → optimal, but expands more nodes.

### Q-Learning
- State = (row, col); action = 4-direction move.
- Reward: -1 per step, +100 on goal, -5 on collision.
- ε-greedy exploration with geometric decay (0.995).
- After training, greedy rollout extracts a path.

## Extending

- **New heuristic** — add a function to `core/heuristics.py` and register it in `HEURISTICS`.
- **New algorithm** — subclass `SearchAgent` in `algorithms/`, implement `solve()`.
- **New scenario** — append to `SCENARIOS` in `benchmarks/runner.py`.

How to Run
bash
### 1. Create project structure (all files above in their respective folders)
### 2. Install dependencies
pip install -r requirements.txt

### 3. Run full benchmark
python main.py

### 4. Or a single demo
python main.py --demo

Sample Output

[Benchmark] Scenario 0: 20x20 | density=0.1 | seed=1
[Benchmark] Scenario 1: 20x20 | density=0.25 | seed=2
...
================================================================================
BENCHMARK SUMMARY
================================================================================
| Algorithm   | Success | Runtime (ms) | Expanded | Path Len | Expanded Density |
|-------------|---------|--------------|----------|----------|------------------|
| A*          | 100.0%  | 2.31         | 112.4    | 37.2     | 31.2%            |
| Dijkstra    | 100.0%  | 8.94         | 341.8    | 37.2     | 92.7%            |
| Q-Learning  | 80.0%   | 421.55       | 24.6     | 39.8     | 6.9%             |
|-------------|---------|--------------|----------|----------|------------------|
Design Highlights
Separation of concerns — core/, algorithms/, metrics/, visualization/, benchmarks/ are decoupled. Each module can be tested independently.

Abstract agent interface — SearchAgent guarantees consistent metrics fields across algorithms, so the tracker treats them uniformly.

Optimality preserved — A* uses Manhattan (admissible for 4-connected grids) and a proper closed set with stale-entry skipping via g_score check.

Q-Learning correctly handles walls — invalid moves incur -5 reward and leave the agent in place, discouraging wall-bumping.

Reproducibility — seeds flow through Grid.random(), QLearningAgent, and MetricsTracker records them in every row.

Professional logging — CSV for spreadsheet analysis, JSON for downstream tooling, plus a human-readable console table.

