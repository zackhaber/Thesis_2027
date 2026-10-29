from time import perf_counter

from env.scenario_generator import Position, Scenario
from .cbs import _neighbors


def _step_towards(start, target, scenario):
    # Greedily choose the valid neighboring cell with minimum Manhattan distance.
    options = [(action, delta) for candidate, action in _neighbors(start, scenario) for delta in [(candidate[0] - start[0], candidate[1] - start[1])]]
    valid = []
    for action, (row_delta, col_delta) in options:
        candidate = (start[0] + row_delta, start[1] + col_delta)
        if 0 <= candidate[0] < scenario.rows and 0 <= candidate[1] < scenario.cols and candidate not in scenario.barrels:
            distance = abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])
            valid.append((distance, action))
    return min(valid)[1] if valid else "stay"


def _advance(position: Position, action: str, scenario: Scenario) -> Position:
    if action == "stay":
        return position
    return next(
        (candidate for candidate, candidate_action in _neighbors(position, scenario)
         if candidate_action == action),
        position,
    )


def voronoi_plan(scenario: Scenario) -> dict:
    # Roll out the greedy controller for a bounded number of grid steps.
    started = perf_counter()
    target = scenario.evaders[0]
    owner = min(range(len(scenario.pursuers)), key=lambda index: (
        abs(scenario.pursuers[index][0] - target[0])
        + abs(scenario.pursuers[index][1] - target[1]),
        index,
    ))
    escape_cells = [
        candidate for candidate, _ in _neighbors(target, scenario)
        if candidate not in scenario.pursuers
    ]
    paths = []
    for index, pursuer in enumerate(scenario.pursuers):
        current = pursuer
        path = []
        assigned_target = target
        if index != owner and escape_cells:
            assigned_target = min(
                escape_cells,
                key=lambda cell: abs(pursuer[0] - cell[0]) + abs(pursuer[1] - cell[1]),
            )
        for _ in range(scenario.rows * scenario.cols):
            action = _step_towards(current, assigned_target, scenario)
            path.append(action)
            current = _advance(current, action, scenario)
            if current == assigned_target:
                break
        paths.append(path)
    return {
        "algorithm": "Voronoi",
        "paths": paths,
        "planning_time": perf_counter() - started,
        "success": bool(paths) and all(paths),
        "owner": owner,
    }
