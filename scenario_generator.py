"""
Generates fair, solvable board layouts for pursuit-evasion games.
Handles grid generation, object placement (barrels, walls, goals),
and reachability verification using Breadth-First Search (BFS).
"""

from dataclasses import dataclass
from random import Random
from env.game_modes import GameMode

# Type aliases for visual clarity
Position = tuple[int, int]
Direction = str

# Default maze wall density (75% wall coverage in MAZE mode)
MAZE_WALL_DENSITY = 0.75
GOAL_MODES = (GameMode.REACH_AVOID, GameMode.MAZE)

# Grid movements: (row_change, col_change)
MOVEMENT_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Used to check if a neighbor cell has a wall pointing back at us
OPPOSITE_DIRECTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass(frozen=True)
class ScenarioSpec:
    """User-selected configuration settings before generating a board."""
    grid_size: int = 20
    pursuers: int = 4
    evaders: int = 1
    pursuer_speed: int = 1
    evader_speed: int = 1
    barrel_density: float = 0.1
    wall_density: float = 0.5
    goals: int = 1
    seed: int = 0


@dataclass(frozen=True)
class Scenario:
    """Immutable finished board layout passed into the environment."""
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


def _manhattan_distance(p1: Position, p2: Position) -> int:
    """Calculate the grid step distance between two positions."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _find_reachable_cells(
    start: Position,
    size: int,
    barrels: frozenset[Position],
    walls: frozenset[tuple[Position, Direction]],
) -> set[Position]:
    """
    Flood-fill (Breadth-First Search) to find every grid cell 
    reachable from `start` without walking through barrels or walls.
    """
    reached = {start}
    queue = [start]

    while queue:
        current = queue.pop()
        
        for direction, (delta_row, delta_col) in MOVEMENT_DELTAS.items():
            neighbor = (current[0] + delta_row, current[1] + delta_col)
            
            # 1. Skip out-of-bounds cells
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
                
            # 2. Skip impassable barrels
            if neighbor in barrels:
                continue

            # 3. Check for directional walls from either side of the edge
            wall_leaving_current = (current, direction) in walls
            wall_entering_neighbor = (neighbor, OPPOSITE_DIRECTION[direction]) in walls
            if wall_leaving_current or wall_entering_neighbor:
                continue

            # 4. Add new valid cell to explored set
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)

    return reached


def _is_valid_scenario(
    pursuers: list[Position],
    evaders: tuple[Position, ...],
    goals: tuple[Position, ...],
    size: int,
    barrels: frozenset[Position],
    walls: frozenset[tuple[Position, Direction]],
) -> bool:
    """
    Verifies that:
    1. Every pursuer, evader, and goal can reach each other.
    2. Evaders do not spawn within 1 step of a pursuer.
    """
    required_locations = set(pursuers) | set(evaders) | set(goals)
    if not required_locations:
        return False

    # Check connectivity from the first entity location
    first_entity = next(iter(required_locations))
    reachable_cells = _find_reachable_cells(first_entity, size, barrels, walls)
    
    if not required_locations.issubset(reachable_cells):
        return False  # Some agent or goal is trapped/unreachable

    # Ensure evader isn't spawned immediately adjacent to any pursuer
    if evaders:
        closest_distance = min(_manhattan_distance(p, evaders[0]) for p in pursuers)
        if closest_distance < 2:
            return False

    return True


def generate_scenario(spec: ScenarioSpec, mode: GameMode = GameMode.PURE_PURSUIT) -> Scenario:
    """
    Generates a deterministic board matching the provided spec and game mode.
    Retries up to 100 times with modified seeds until a valid board is generated.
    """
    size = spec.grid_size
    barrel_density = spec.barrel_density
    wall_density = spec.wall_density
    
    # Mode overrides
    if mode == GameMode.MAZE:
        barrel_density = 0.0
        wall_density = MAZE_WALL_DENSITY

    # Retry loop for deterministic generation
    for attempt in range(100):
        rng = Random(spec.seed + attempt)
        
        # Create a list of all possible grid cells and shuffle them
        available_cells = [(row, col) for row in range(size) for col in range(size)]
        rng.shuffle(available_cells)

        # 1. Place Pursuers
        pursuer_positions = [available_cells.pop() for _ in range(spec.pursuers)]

        # 2. Place Barrels (if applicable)
        barrel_count = min(len(available_cells), int(size * size * barrel_density))
        barrel_list = [available_cells.pop() for _ in range(barrel_count)]
        barrels = frozenset(barrel_list)

        # 3. Place Directional Walls
        wall_count = min(len(available_cells), int(size * size * wall_density))
        directions = ("up", "down", "left", "right")
        wall_list = [(available_cells.pop(), rng.choice(directions)) for _ in range(wall_count)]
        walls = frozenset(wall_list)

        # 4. Place Goals and Evaders
        if mode in GOAL_MODES:
            # Pick goal positions from remaining open cells
            open_cells = [cell for cell in available_cells if cell not in barrels]
            goal_count = max(1, spec.goals)
            goals = tuple(open_cells[:goal_count])

            # Place evader on open cells furthest from goals
            evader_candidates = [c for c in open_cells if c not in goals]
            evader_candidates.sort(
                key=lambda cell: min(_manhattan_distance(cell, g) for g in goals),
                reverse=True
            )
            evader_positions = tuple(evader_candidates[:spec.evaders])
        else:
            goals = ()
            evader_positions = tuple(available_cells.pop() for _ in range(spec.evaders))

        # 5. Validate connectivity and fairness
        if _is_valid_scenario(pursuer_positions, evader_positions, goals, size, barrels, walls):
            return Scenario(
                rows=size,
                cols=size,
                pursuers=tuple(pursuer_positions),
                evaders=evader_positions,
                barrels=barrels,
                walls=walls,
                goals=goals,
                pursuer_speed=spec.pursuer_speed,
                evader_speed=spec.evader_speed,
                seed=spec.seed + attempt,
            )

    raise ValueError(f"Could not generate a valid scenario for seed {spec.seed} after 100 attempts.")