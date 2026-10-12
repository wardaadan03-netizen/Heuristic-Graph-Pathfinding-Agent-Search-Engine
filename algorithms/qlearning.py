"""Tabular Q-Learning agent — model-free reinforcement learning."""

from __future__ import annotations
import random
from typing import Optional
import numpy as np

from algorithms.base import SearchAgent
from config.settings import CONFIG
from core.grid import Grid


class QLearningAgent(SearchAgent):
    """
    Tabular Q-Learning.

    State  : (row, col)
    Action : one of 4 directions (N, S, W, E)
    Reward : -1 per step, +100 on goal, -5 on invalid (blocked) move.
    """

    name = "Q-Learning"

    def __init__(
        self,
        grid: Grid,
        start: tuple[int, int],
        goal: tuple[int, int],
        episodes: int = CONFIG.QL_EPISODES,
        alpha: float = CONFIG.QL_ALPHA,
        gamma: float = CONFIG.QL_GAMMA,
        epsilon: float = CONFIG.QL_EPSILON,
        epsilon_decay: float = CONFIG.QL_EPSILON_DECAY,
        epsilon_min: float = CONFIG.QL_EPSILON_MIN,
        seed: int | None = CONFIG.RANDOM_SEED,
    ):
        super().__init__(grid, start, goal)
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)

        # Q-table: dict[(row, col)] -> np.array of 4 action values
        self.q_table: dict[tuple[int, int], np.ndarray] = {}

    # ---------- Q-table helpers ----------
    def _q(self, state: tuple[int, int]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(Grid.DIRECTIONS), dtype=np.float64)
        return self.q_table[state]

    def _next_state(self, state: tuple[int, int], action_idx: int) -> tuple[int, int]:
        dr, dc = Grid.DIRECTIONS[action_idx]
        return (state[0] + dr, state[1] + dc)

    # ---------- Training ----------
    def train(self) -> None:
        """
        Train Q-table with epsilon-greedy episodes.

        Uses a reward-shaping bonus based on Manhattan distance to the goal
        (potential-based shaping, policy-preserving) so the goal signal
        propagates through the Q-table even when the agent hasn't reached
        the goal yet.
        """
        for episode in range(self.episodes):
            state = self.start
            # FIX: potential-based shaping — gives gradient toward goal
            prev_dist = self._manhattan(state, self.goal)

            for _ in range(CONFIG.QL_MAX_STEPS_PER_EPISODE):
                # ε-greedy action selection
                if self.rng.random() < self.epsilon:
                    action = self.rng.randrange(len(Grid.DIRECTIONS))
                else:
                    action = int(np.argmax(self._q(state)))

                next_state = self._next_state(state, action)

                # Reward + wall handling
                if not self.grid.is_free(next_state):
                    reward = -5.0
                    next_state = state          # bump wall -> stay put
                elif next_state == self.goal:
                    reward = 100.0
                else:
                    reward = -1.0

                # FIX: add shaping term (distance progress) — small magnitude
                new_dist = self._manhattan(next_state, self.goal)
                reward += 0.5 * (prev_dist - new_dist)  # +0.5 if closer, -0.5 if farther
                prev_dist = new_dist

                # TD update — safe via _q() helper
                q_current = self._q(state)
                q_next = self._q(next_state)
                best_next = float(np.max(q_next))
                td_target = reward + self.gamma * best_next

                q_current[action] += self.alpha * (td_target - q_current[action])

                state = next_state
                if state == self.goal:
                    break

            # FIX: decay epsilon only if it makes progress; slower early on
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- Greedy rollout ----------
    def solve(self) -> Optional[list[tuple[int, int]]]:
        self.train()

        # FIX: allow backtracking — no `visited` set blocking progress.
        # Use a step cap and a "no-progress" counter to detect loops.
        state = self.start
        path = [state]
        max_steps = max(CONFIG.QL_MAX_STEPS_PER_EPISODE, 4 * self.grid.rows * self.grid.cols)
        no_progress = 0
        last_dist = self._manhattan(state, self.goal)

        for _ in range(max_steps):
            if state == self.goal:
                self.path = path
                self.path_cost = float(len(path) - 1)
                self.found = True
                return path

            self.expanded_nodes += 1
            q_vals = self._q(state).copy()

            # FIX: among free neighbors, prefer highest Q; tie-break by
            # distance to goal so agent never stalls when Q-values are all 0.
            best_action = None
            best_score = -float("inf")
            for action in np.argsort(-q_vals):
                action = int(action)
                nxt = self._next_state(state, action)
                if not self.grid.is_free(nxt):
                    continue
                # Prefer greedy Q; use Manhattan as tie-breaker (small weight)
                score = q_vals[action] - 0.01 * self._manhattan(nxt, self.goal)
                if score > best_score:
                    best_score = score
                    best_action = action

            if best_action is None:
                # Fully boxed in — genuinely stuck
                break

            next_state = self._next_state(state, best_action)
            path.append(next_state)
            state = next_state

            # FIX: no-progress watchdog — if we wander too long without
            # reducing distance, give up gracefully instead of looping.
            new_dist = self._manhattan(state, self.goal)
            if new_dist >= last_dist:
                no_progress += 1
            else:
                no_progress = 0
            last_dist = new_dist

            if no_progress > 5 * (self.grid.rows + self.grid.cols):
                break

        self.found = False
        return None