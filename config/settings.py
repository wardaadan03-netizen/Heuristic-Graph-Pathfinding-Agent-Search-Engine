""" Global configuration for the pathfinding engine"""

from dataclasses import dataclass 

@dataclass(frozen=True)
class config:
    """ Engine-wide configuration constants."""
    DEFAULT_GRID_SIZE:int=30
    DEFAULT_OBSTACLE_DENSITY:float=0.25
    RANDOM_SEED:int=42

    # Q-Learning hyperparameters
    QL_EPISODES: int = 1500              # was 400
    QL_ALPHA: float = 0.20               # was 0.15 — faster learning
    QL_GAMMA: float = 0.97               # was 0.95 — value long horizons more
    QL_EPSILON: float = 1.0
    QL_EPSILON_DECAY: float = 0.995      # keep
    QL_EPSILON_MIN: float = 0.05
    QL_MAX_STEPS_PER_EPISODE: int = 600  # was 400 — bigger grids need more steps

CONFIG=config()
