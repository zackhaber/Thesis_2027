from dataclasses import dataclass
from random import Random

from .game_modes import GameMode

Position = tuple[int, int]
Direction = str
FIXED_SPEED = 1
MAZE_WALL_DENSITY = 0.75
GOAL_MODES = (GameMode.REACH_AVOID, GameMode.MAZE)
MOVEMENT_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_DIRECTION = {"up": "down", "down": "up", "left": "right", "right": "left"}

LEVEL_PRESETS = {
    1: {"grid_size": 5, "pursuers": 2, "barrel_density": 0.0, "wall_density": 0.0},
    2: {"grid_size": 10, "pursuers": 3, "barrel_density": 0.04, "wall_density": 0.05},
    3: {"grid_size": 20, "pursuers": 4, "barrel_density": 0.08, "wall_density": 0.12},
}

@dataclass(frozen=True)
class DifficultySpec:
    # Describe a requested level before any random cells are generated.
    level: int = 1
    grid_size: int | None = None
    pursuers: int | None = None
    evaders: int = 1
    pursuer_speed: int = 1
    evader_speed: int = 1
    obstacle_density: float | None = None
    barrel_density: float | None = None
    wall_density: float | None = None
    goals: int = 1
    seed: int = 7

@dataclass(frozen=True)
class Scenario:
    # Store the completed immutable board and its movement parameters.
    rows: int
    cols: int
    pursuers: tuple[Position, ...]
    evaders: tuple[Position, ...]
    barrels: frozenset[Position]
    walls: frozenset[tuple[Position, Direction]]
    goals: tuple[Position, ...]
    pursuer_speed: int
    evader_speed: int
    seed: int

    @property
    def obstacles(self) -> frozenset[Position]:
        # Preserve the older obstacle name as an alias for barrel cells.
        return self.barrels


def _preset(spec: DifficultySpec) -> tuple[int, float, float, int, int]:
    # Resolve defaults while accepting obstacle_density as a legacy barrel alias.
    preset = LEVEL_PRESETS.get(spec.level, LEVEL_PRESETS[2])
    size = spec.grid_size or preset["grid_size"]
    barrel_density = spec.barrel_density
    if barrel_density is None:
        barrel_density = spec.obstacle_density
    if barrel_density is None:
        barrel_density = preset["barrel_density"]
    wall_density = spec.wall_density if spec.wall_density is not None else preset["wall_density"]
    return size, barrel_density, wall_density, FIXED_SPEED, FIXED_SPEED


def _distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _has_open_neighbor(position: Position, size: int, barrels: frozenset[Position], walls: frozenset[tuple[Position, Direction]]) -> bool:
    # Maze repair uses this to reject cells sealed by barrels, borders, or walls.
    for direction, (row_delta, col_delta) in MOVEMENT_DELTAS.items():
        neighbor = (position[0] + row_delta, position[1] + col_delta)
        if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
            continue
        if neighbor in barrels:
            continue
        if (position, direction) in walls or (neighbor, OPPOSITE_DIRECTION[direction]) in walls:
            continue
        return True
    return False


def generate_level(spec: DifficultySpec, mode: GameMode = GameMode.PURE_PURSUIT) -> Scenario:
    # Build a reproducible board by shuffling cells, placing agents, then obstacles.
    size, barrel_density, wall_density, pursuer_speed, evader_speed = _preset(spec)
    if mode == GameMode.MAZE:
        # Maze mode replaces barrels with a much denser wall layout.
        wall_density = MAZE_WALL_DENSITY
    rng = Random(spec.seed)
    cells = [(row, col) for row in range(size) for col in range(size)]
    rng.shuffle(cells)
    preset = LEVEL_PRESETS.get(spec.level, LEVEL_PRESETS[2])
    pursuer_count = spec.pursuers if spec.pursuers is not None else preset["pursuers"]
    pursuer_positions = [cells.pop() for _ in range(pursuer_count)]
    barrel_count = min(len(cells), int(size * size * barrel_density))
    barrels = frozenset(cells.pop() for _ in range(barrel_count)) if mode != GameMode.MAZE else frozenset()
    wall_count = min(len(cells), int(size * size * wall_density))
    directions = ("up", "down", "left", "right")
    walls = frozenset((cells.pop(), rng.choice(directions)) for _ in range(wall_count))
    if mode == GameMode.MAZE:
        # Move any isolated pursuer to a cell with at least one traversable neighbor.
        for index, pursuer in enumerate(pursuer_positions):
            if _has_open_neighbor(pursuer, size, barrels, walls):
                continue
            replacement_index = next(
                (cell_index for cell_index, cell in enumerate(cells) if _has_open_neighbor(cell, size, barrels, walls)),
                None,
            )
            if replacement_index is not None:
                pursuer_positions[index], cells[replacement_index] = cells[replacement_index], pursuer
    if mode in GOAL_MODES:
        # Place goals first, then start evaders in cells farthest from those goals.
        goals = tuple(cell for cell in cells if cell not in barrels)[: max(1, spec.goals)]
        if not goals:
            goals = (next(cell for cell in cells if cell not in barrels),)
        available_evader_cells = [cell for cell in cells if cell not in goals and cell not in barrels]
        available_evader_cells.sort(key=lambda cell: min(_distance(cell, goal) for goal in goals), reverse=True)
        evader_positions = tuple(available_evader_cells[: spec.evaders])
    else:
        # Pure pursuit has no goal condition, so remaining cells become evader starts.
        goals = ()
        evader_positions = tuple(cells.pop() for _ in range(spec.evaders))
    return Scenario(size, size, tuple(pursuer_positions), evader_positions, barrels, walls, goals, pursuer_speed, evader_speed, spec.seed)
