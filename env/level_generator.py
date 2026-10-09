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


def _reachable(start: Position, size: int, barrels: frozenset[Position], walls: frozenset[tuple[Position, Direction]]) -> set[Position]:
    reached = {start}
    pending = [start]
    while pending:
        current = pending.pop()
        for direction, (row_delta, col_delta) in MOVEMENT_DELTAS.items():
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
            if neighbor in barrels or (current, direction) in walls or (neighbor, OPPOSITE_DIRECTION[direction]) in walls:
                continue
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _is_valid_scenario(
    pursuers: list[Position],
    evaders: tuple[Position, ...],
    goals: tuple[Position, ...],
    size: int,
    barrels: frozenset[Position],
    walls: frozenset[tuple[Position, Direction]],
) -> bool:
    required = set(pursuers) | set(evaders) | set(goals)
    if not required:
        return False
    reachable = _reachable(next(iter(required)), size, barrels, walls)
    if not required.issubset(reachable):
        return False
    if evaders and min(_distance(pursuer, evaders[0]) for pursuer in pursuers) < 2:
        return False
    return True


def generate_level(spec: DifficultySpec, mode: GameMode = GameMode.PURE_PURSUIT) -> Scenario:
    # Retry from deterministic derived seeds until the board is connected and fair.
    size, barrel_density, wall_density, pursuer_speed, evader_speed = _preset(spec)
    if mode == GameMode.MAZE:
        # Maze mode replaces barrels with a much denser wall layout.
        wall_density = MAZE_WALL_DENSITY
    preset = LEVEL_PRESETS.get(spec.level, LEVEL_PRESETS[2])
    pursuer_count = spec.pursuers if spec.pursuers is not None else preset["pursuers"]
    for attempt in range(100):
        rng = Random(spec.seed + attempt)
        cells = [(row, col) for row in range(size) for col in range(size)]
        rng.shuffle(cells)
        pursuer_positions = [cells.pop() for _ in range(pursuer_count)]
        barrel_count = min(len(cells), int(size * size * barrel_density))
        barrels = frozenset(cells.pop() for _ in range(barrel_count)) if mode != GameMode.MAZE else frozenset()
        wall_count = min(len(cells), int(size * size * wall_density))
        directions = ("up", "down", "left", "right")
        walls = frozenset((cells.pop(), rng.choice(directions)) for _ in range(wall_count))
        if mode in GOAL_MODES:
            goals = tuple(cell for cell in cells if cell not in barrels)[: max(1, spec.goals)]
            available_evader_cells = [cell for cell in cells if cell not in goals and cell not in barrels]
            available_evader_cells.sort(key=lambda cell: min(_distance(cell, goal) for goal in goals), reverse=True)
            evader_positions = tuple(available_evader_cells[: spec.evaders])
        else:
            goals = ()
            evader_positions = tuple(cells.pop() for _ in range(spec.evaders))
        if _is_valid_scenario(pursuer_positions, evader_positions, goals, size, barrels, walls):
            return Scenario(size, size, tuple(pursuer_positions), evader_positions, barrels, walls, goals, pursuer_speed, evader_speed, spec.seed + attempt)
    raise ValueError(f"Could not generate a valid level for seed {spec.seed}")
