from collections import deque
from itertools import product
from time import perf_counter

from env.scenario_generator import Position, Scenario
from .cbs import _neighbors


def _distance_map(target: Position, scenario: Scenario) -> dict[Position, int]:
    distances = {target: 0}
    pending = deque([target])
    while pending:
        current = pending.popleft()
        for candidate, _ in _neighbors(current, scenario):
            if candidate not in distances:
                distances[candidate] = distances[current] + 1
                pending.append(candidate)
    return distances


def _options(position: Position, distances: dict[Position, int], scenario: Scenario):
    options = [(position, "stay")]
    options.extend(_neighbors(position, scenario))
    return sorted(options, key=lambda item: (distances.get(item[0], float("inf")), item[1]))


def _valid_joint_move(
    current: tuple[Position, ...],
    proposed: tuple[Position, ...],
) -> bool:
    if len(set(proposed)) != len(proposed):
        return False
    return all(
        current[first] != proposed[second]
        or current[second] != proposed[first]
        for first in range(len(current))
        for second in range(first + 1, len(current))
    )


def mstar_plan(scenario: Scenario) -> dict:
    """Plan with decoupled BFS routes and joint expansion only at collisions."""
    started = perf_counter()
    if not scenario.evaders:
        return {"algorithm": "M*", "paths": [], "planning_time": 0.0, "success": False}
    target = scenario.evaders[0]
    distance_maps = [_distance_map(target, scenario) for _ in scenario.pursuers]
    starts = tuple(scenario.pursuers)
    queue = deque([(starts, [[] for _ in starts])])
    visited = {starts}
    horizon = scenario.rows * scenario.cols * 2
    while queue:
        positions, paths = queue.popleft()
        if any(position == target for position in positions):
            return {
                "algorithm": "M*",
                "paths": paths,
                "planning_time": perf_counter() - started,
                "success": True,
            }
        if max((len(path) for path in paths), default=0) >= horizon:
            continue
        choices = [
            _options(position, distance_maps[index], scenario)
            for index, position in enumerate(positions)
        ]
        preferred = tuple(options[0][0] for options in choices)
        combinations = product(*choices) if not _valid_joint_move(positions, preferred) else [tuple(options[0] for options in choices)]
        for combination in combinations:
            next_positions = tuple(item[0] for item in combination)
            if not _valid_joint_move(positions, next_positions) or next_positions in visited:
                continue
            visited.add(next_positions)
            next_paths = [path + [combination[index][1]] for index, path in enumerate(paths)]
            queue.append((next_positions, next_paths))
    return {
        "algorithm": "M*",
        "paths": [[] for _ in scenario.pursuers],
        "planning_time": perf_counter() - started,
        "success": False,
    }
