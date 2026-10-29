from collections import deque
from dataclasses import dataclass
from time import perf_counter

from env.scenario_generator import Position, Scenario

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
        if inside_grid and candidate not in scenario.barrels and not blocked_by_wall:
            yield candidate, action


@dataclass(frozen=True)
class _Constraint:
    agent: int
    time: int
    position: Position
    previous: Position | None = None


def _shortest_path(
    start: Position,
    targets: set[Position],
    scenario: Scenario,
    constraints: set[_Constraint],
) -> list[str]:
    """Find a shortest time-expanded route satisfying one agent's constraints."""
    horizon = max(1, scenario.rows * scenario.cols * 2)
    queue = deque([(start, 0, [])])
    visited = {(start, 0)}
    while queue:
        position, time, path = queue.popleft()
        if position in targets:
            return path
        if time >= horizon:
            continue
        candidates = [(position, "stay")]
        candidates.extend(_neighbors(position, scenario))
        for candidate, action in candidates:
            next_time = time + 1
            if any(
                item.time == next_time and item.position == candidate
                for item in constraints
            ):
                continue
            if any(
                item.time == next_time
                and item.previous == candidate
                and item.position == position
                for item in constraints
            ):
                continue
            state = (candidate, next_time)
            if state not in visited:
                visited.add(state)
                queue.append((candidate, next_time, path + [action]))
    return []


def _position_at(path: list[str], start: Position, time: int, scenario: Scenario) -> Position:
    position = start
    for action in path[:time]:
        if action == "stay":
            continue
        position = next(
            (candidate for candidate, candidate_action in _neighbors(position, scenario)
             if candidate_action == action),
            position,
        )
    return position


def _find_conflict(paths: list[list[str]], scenario: Scenario):
    starts = scenario.pursuers
    target = scenario.evaders[0] if scenario.evaders else None
    arrival_times = [
        next(
            (time for time in range(len(path) + 1)
             if _position_at(path, start, time, scenario) == target),
            len(path),
        )
        for path, start in zip(paths, starts)
    ]
    horizon = min(arrival_times, default=0)
    for time in range(1, horizon + 1):
        for first in range(len(paths)):
            first_position = _position_at(paths[first], starts[first], time, scenario)
            first_previous = _position_at(paths[first], starts[first], time - 1, scenario)
            for second in range(first + 1, len(paths)):
                second_position = _position_at(paths[second], starts[second], time, scenario)
                second_previous = _position_at(paths[second], starts[second], time - 1, scenario)
                if first_position == second_position:
                    return first, second, time, first_position, second_position
                if first_previous == second_position and second_previous == first_position:
                    return first, second, time, first_position, second_position
    return None


def cbs_plan(scenario: Scenario) -> dict:
    started = perf_counter()
    targets = {scenario.evaders[0]} if scenario.evaders else set()
    root_paths = [
        _shortest_path(pursuer, targets, scenario, set())
        for pursuer in scenario.pursuers
    ]
    open_nodes: list[tuple[int, int, list[list[str]], tuple[frozenset[_Constraint], ...]]] = []
    constraints = tuple(frozenset() for _ in scenario.pursuers)
    open_nodes.append((sum(map(len, root_paths)), 0, root_paths, constraints))
    node_id = 1
    while open_nodes:
        open_nodes.sort(key=lambda node: (node[0], node[1]))
        _, _, paths, node_constraints = open_nodes.pop(0)
        conflict = _find_conflict(paths, scenario)
        if conflict is None:
            return {
                "algorithm": "CBS",
                "paths": paths,
                "planning_time": perf_counter() - started,
                "success": bool(paths) and all(paths),
            }
        first, second, time, first_position, second_position = conflict
        for agent, position, previous in (
            (
                first,
                first_position,
                second_position if first_position != second_position else None,
            ),
            (
                second,
                second_position,
                first_position if first_position != second_position else None,
            ),
        ):
            updated_constraints = list(node_constraints)
            constraint = _Constraint(agent, time, position, previous)
            updated_constraints[agent] = frozenset((*updated_constraints[agent], constraint))
            replanned = _shortest_path(
                scenario.pursuers[agent],
                targets,
                scenario,
                set(updated_constraints[agent]),
            )
            if not replanned:
                continue
            updated_paths = list(paths)
            updated_paths[agent] = replanned
            open_nodes.append((
                sum(map(len, updated_paths)),
                node_id,
                updated_paths,
                tuple(updated_constraints),
            ))
            node_id += 1
    return {
        "algorithm": "CBS",
        "paths": root_paths,
        "planning_time": perf_counter() - started,
        "success": False,
    }
