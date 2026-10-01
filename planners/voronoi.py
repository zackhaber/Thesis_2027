from time import perf_counter

from env.level_generator import Scenario


def _step_towards(start, target, scenario):
    # Greedily choose the valid neighboring cell with minimum Manhattan distance.
    options = [("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))]
    valid = []
    for action, (row_delta, col_delta) in options:
        candidate = (start[0] + row_delta, start[1] + col_delta)
        if 0 <= candidate[0] < scenario.rows and 0 <= candidate[1] < scenario.cols and candidate not in scenario.obstacles:
            distance = abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])
            valid.append((distance, action))
    return min(valid)[1] if valid else "up"


def voronoi_plan(scenario: Scenario) -> dict:
    # Roll out the greedy controller for a bounded number of grid steps.
    started = perf_counter()
    target = scenario.evaders[0]
    paths = []
    for pursuer in scenario.pursuers:
        current = pursuer
        path = []
        for _ in range(scenario.rows * scenario.cols):
            action = _step_towards(current, target, scenario)
            path.append(action)
            if action == "up": current = (current[0] - 1, current[1])
            elif action == "down": current = (current[0] + 1, current[1])
            elif action == "left": current = (current[0], current[1] - 1)
            elif action == "right": current = (current[0], current[1] + 1)
            if current == target: break
        paths.append(path)
    return {"algorithm": "Voronoi", "paths": paths, "planning_time": perf_counter() - started, "success": any(paths)}
