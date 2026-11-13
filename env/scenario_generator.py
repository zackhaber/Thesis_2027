"""
Generates fair, solvable board layouts for pursuit-evasion games.
Handles grid generation, object placement (barrels, walls, goals),
and reachability verification using Breadth-First Search (BFS).
"""

from collections import deque
from dataclasses import dataclass
from random import Random
from .game_modes import GameMode

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

# User-selected configuration settings before generating a board.
@dataclass(frozen=True)
class ScenarioSpec:
    grid_size: int = 20
    pursuers: int = 4
    evaders: int = 1
    pursuer_speed: int = 1
    evader_speed: int = 1
    barrel_density: float = 0.1
    wall_density: float = 0.5
    goals: int = 1
    seed: int = 0
    defender_radius: int = 3

# Immutable finished board layout passed into the environment.
@dataclass(frozen=True)
class Scenario:
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
    bases: tuple[Position, ...] = ()

# Calculate the grid step distance between two positions.
def _manhattan_distance(p1: Position, p2: Position) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Find all cells reachable from a starting position.
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

# Check if a generated scenario is valid.
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


def _ctf_distances(
    start: Position,
    size: int,
    barrels: frozenset[Position],
    walls: frozenset[tuple[Position, Direction]],
) -> dict[Position, int]:
    distances = {start: 0}
    pending = deque([start])
    while pending:
        position = pending.popleft()
        for direction, (dr, dc) in MOVEMENT_DELTAS.items():
            neighbor = (position[0] + dr, position[1] + dc)
            if (
                not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size)
                or neighbor in barrels
                or neighbor in distances
                or (position, direction) in walls
                or (neighbor, OPPOSITE_DIRECTION[direction]) in walls
            ):
                continue
            distances[neighbor] = distances[position] + 1
            pending.append(neighbor)
    return distances


def _generate_ctf_scenario(spec: ScenarioSpec) -> Scenario:
    if spec.grid_size < 2 or spec.pursuers < 1 or spec.evaders < 1:
        raise ValueError("CTF requires a grid of at least 2 and at least one agent per team.")
    if spec.defender_radius < 0:
        raise ValueError("defender_radius must be non-negative.")
    if not 0 <= spec.barrel_density <= 1 or not 0 <= spec.wall_density <= 1:
        raise ValueError("CTF obstacle densities must be between 0 and 1.")

    size = spec.grid_size
    for attempt in range(100):
        rng = Random(spec.seed + attempt)
        cells = [(row, col) for row in range(size) for col in range(size)]
        rng.shuffle(cells)
        flag = cells.pop()
        barrel_count = min(len(cells), int(size * size * spec.barrel_density))
        barrels = frozenset(cells.pop() for _ in range(barrel_count))
        wall_count = min(len(cells), int(size * size * spec.wall_density))
        walls = frozenset(
            (cells.pop(), rng.choice(("up", "down", "left", "right")))
            for _ in range(wall_count)
        )
        distances = _ctf_distances(flag, size, barrels, walls)
        free_cells = [
            (row, col)
            for row in range(size)
            for col in range(size)
            if (row, col) in distances and (row, col) != flag
        ]
        if not free_cells:
            continue
        rng.shuffle(free_cells)
        base = max(free_cells, key=distances.__getitem__)
        defenders = [
            cell for cell in free_cells
            if cell != base
            and distances[cell] <= spec.defender_radius
            and _manhattan_distance(cell, base) >= 2
        ][:spec.pursuers]
        if len(defenders) != spec.pursuers:
            continue
        attackers = (base,) * spec.evaders
        if _is_valid_scenario(defenders, attackers, (flag, base), size, barrels, walls):
            return Scenario(
                rows=size,
                cols=size,
                pursuers=tuple(defenders),
                evaders=attackers,
                barrels=barrels,
                walls=walls,
                goals=(flag,),
                pursuer_speed=spec.pursuer_speed,
                evader_speed=spec.evader_speed,
                seed=spec.seed + attempt,
                bases=(base,),
            )
    raise ValueError(f"Could not generate a valid scenario for seed {spec.seed} after 100 attempts.")


"""
    Generates a deterministic board matching the provided spec and game mode.
    Retries up to 100 times with modified seeds until a valid board is generated.
    """
def generate_scenario(spec: ScenarioSpec, mode: GameMode = GameMode.PURE_PURSUIT) -> Scenario:
    """Generate a deterministic scenario; CTF uses one shared attacker base."""
    if mode == GameMode.CAPTURE_THE_FLAG:
        return _generate_ctf_scenario(spec)
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

        # Place Pursuers
        pursuer_positions = [available_cells.pop() for _ in range(spec.pursuers)]

        # Place Barrels (if applicable)
        barrel_count = min(len(available_cells), int(size * size * barrel_density))
        barrel_list = [available_cells.pop() for _ in range(barrel_count)]
        barrels = frozenset(barrel_list)

        # Place Directional Walls
        wall_count = min(len(available_cells), int(size * size * wall_density))
        directions = ("up", "down", "left", "right")
        wall_list = [(available_cells.pop(), rng.choice(directions)) for _ in range(wall_count)]
        walls = frozenset(wall_list)

        # Place Goals and Evaders
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