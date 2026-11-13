from enum import Enum

# These modes determine the evader's terminal objective.
class GameMode(str, Enum):
    PURE_PURSUIT = "pure_pursuit"
    REACH_AVOID = "reach_avoid"
    MAZE = "maze"
    CAPTURE_THE_FLAG = "capture_the_flag"
