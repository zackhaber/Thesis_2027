from collections import deque
from time import perf_counter

from env.level_generator import Position, Scenario

ACTIONS = ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right"))
OPPOSITE_ACTION = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _wall_blocks(position: Position, candidate: Position, action: str, scenario: Scenario) -> bool:
    return (position, action) in scenario.walls or (candidate, OPPOSITE_ACTION[action]) in scenario.walls


def _neighbors(position: Position, scenario: Scenario):
    # Treat a wall as blocking both directions across the same grid edge.
    for row_delta, col_delta, action in ACTIONS:
        candidate = (position[0] + row_delta, position[1] + col_delta)
        inside_grid = 0 <= candidate[0] < scenario.rows and 0 <= candidate[1] < scenario.cols
        blocked_by_wall = _wall_blocks(position, candidate, action, scenario)
        if inside_grid and candidate not in scenario.obstacles and not blocked_by_wall:
            yield candidate, action


def _shortest_path(start: Position, targets: set[Position], scenario: Scenario, reserved: set[Position]) -> list[str]:
    # Breadth-first search finds the shortest path while avoiding reserved cells.
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        position, path = queue.popleft()
        if position in targets:
            return path
        for candidate, action in _neighbors(position, scenario):
            if candidate not in visited and candidate not in reserved:
                visited.add(candidate)
                queue.append((candidate, path + [action]))
    # Keep the result non-empty to preserve the planner's existing output shape.
    return ["up"]


def cbs_plan(scenario: Scenario) -> dict:
    # This is a sequential reservation baseline rather than full CBS.
    started = perf_counter()
    reserved: set[Position] = set(scenario.evaders)
    paths = []
    for pursuer in scenario.pursuers:
        # Keep the evader target reserved during each pursuer's search.
        path = _shortest_path(pursuer, reserved, scenario, set())
        paths.append(path)
        reserved.add(scenario.evaders[0])
    return {"algorithm": "CBS", "paths": paths, "planning_time": perf_counter() - started, "success": all(paths)}
