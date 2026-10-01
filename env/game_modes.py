from enum import Enum

class GameMode(str, Enum):
    PURE_PURSUIT = "pure_pursuit"
    REACH_AVOID = "reach_avoid"
    MAZE = "maze"
