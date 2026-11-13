"""Pure CTF strategies; all planners receive the same defender assignments."""

from collections import deque
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from typing import Callable

from env.env import ACTION_ORDER, MOVES, OPPOSITE_ACTION, Action
from env.knowledge import EvaderBelief, PursuerBelief, attacker_view, explore_action
from env.scenario_generator import Position, Scenario
from typing import Protocol


@dataclass(frozen=True)
class CTFConfig:
    guards: int = 1
    guard_radius: int = 2


@dataclass(frozen=True)
class FlagState:
    carrier: int | None = None


class SearchPolicy(Protocol):
    """Search controller receiving only a team belief and controlled positions."""
    is_active: bool

    def act(self, belief: PursuerBelief, positions: tuple[Position, ...]) -> list[Action]: ...


class FrontierSearch:
    """Default search policy using the shared frontier exploration helper."""
    is_active = True

    def act(self, belief: PursuerBelief, positions: tuple[Position, ...]) -> list[Action]:
        return [explore_action(belief, position) for position in positions]


def _move(view: Scenario, position: Position, action: Action, speed: int) -> Position:
    row, col = position
    dr, dc = MOVES[action]
    for _ in range(speed):
        candidate = (row + dr, col + dc)
        if (
            not (0 <= candidate[0] < view.rows and 0 <= candidate[1] < view.cols)
            or candidate in view.barrels
            or ((row, col), action) in view.walls
            or (candidate, OPPOSITE_ACTION[action]) in view.walls
        ):
            break
        row, col = candidate
    return row, col


def _routes(
    view: Scenario,
    start: Position,
    speed: int,
    blocked: set[Position] | None = None,
    avoid_first: Position | None = None,
) -> tuple[dict[Position, int], dict[Position, tuple[Position, Action]]]:
    distances = {start: 0}
    previous: dict[Position, tuple[Position, Action]] = {}
    pending = deque([start])
    while pending:
        position = pending.popleft()
        for action in ACTION_ORDER[1:]:
            candidate = _move(view, position, action, speed)
            if (
                candidate == position or candidate in distances
                or (blocked and candidate in blocked)
                or (position == start and candidate == avoid_first)
            ):
                continue
            distances[candidate] = distances[position] + 1
            previous[candidate] = (position, action)
            pending.append(candidate)
    return distances, previous


def _path(
    start: Position, target: Position, previous: dict[Position, tuple[Position, Action]]
) -> list[Position]:
    if target != start and target not in previous:
        return []
    path = [target]
    while path[-1] != start:
        path.append(previous[path[-1]][0])
    path.reverse()
    return path


def _first_action(
    start: Position, target: Position, previous: dict[Position, tuple[Position, Action]]
) -> Action:
    if target == start or target not in previous:
        return "stay"
    current = target
    while previous[current][0] != start:
        current = previous[current][0]
    return previous[current][1]


def _danger_routes(
    view: Scenario, start: Position, defenders: set[Position], danger: set[Position],
    avoid_first: Position | None = None,
) -> tuple[dict[Position, int], dict[Position, tuple[Position, Action]]]:
    """Find a minimum-risk route, while still charging for each step."""
    distances = {start: 0}
    previous: dict[Position, tuple[Position, Action]] = {}
    order = count()
    pending = [(0, next(order), start)]
    while pending:
        distance, _, position = heappop(pending)
        if distance != distances[position]:
            continue
        for action in ACTION_ORDER[1:]:
            candidate = _move(view, position, action, view.evader_speed)
            if candidate == position or (position == start and candidate == avoid_first):
                continue
            cost = 1 + (20 if candidate in defenders else 4 if candidate in danger else 0)
            new_distance = distance + cost
            if new_distance < distances.get(candidate, float("inf")):
                distances[candidate] = new_distance
                previous[candidate] = (position, action)
                heappush(pending, (new_distance, next(order), candidate))
    return distances, previous


def attacker_action(
    view: Scenario,
    position: Position,
    defenders: tuple[Position, ...] | set[Position],
    carrying: bool,
    flag_position: Position | None = None,
    previous_position: Position | None = None,
) -> Action:
    """Head for the flag or a base while avoiding defender-adjacent cells."""
    if not view.goals or not view.bases:
        raise ValueError("CTF strategy requires a flag home and at least one base.")
    targets = view.bases if carrying else (flag_position if flag_position is not None else view.goals[0],)
    danger = {
        (defender[0] + dr, defender[1] + dc)
        for defender in defenders
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    }
    for avoided in (previous_position, None) if previous_position is not None else (None,):
        distances, previous = _routes(view, position, view.evader_speed, danger, avoided)
        reachable_targets = [target for target in targets if target in distances]
        if reachable_targets:
            target = min(reachable_targets, key=lambda cell: (distances[cell], cell))
            return _first_action(position, target, previous)

    for avoided in (previous_position, None) if previous_position is not None else (None,):
        risk_distances, risk_previous = _danger_routes(view, position, set(defenders), danger, avoided)
        reachable_targets = [target for target in targets if target in risk_distances]
        if reachable_targets:
            target = min(reachable_targets, key=lambda cell: (risk_distances[cell], cell))
            return _first_action(position, target, risk_previous)
    target = targets[0]
    target_distances, _ = _routes(view, target, 1)
    options = [
        (action, _move(view, position, action, view.evader_speed))
        for action in ACTION_ORDER[1:]
    ]
    options = [(action, cell) for action, cell in options if cell != position]
    if not options:
        return "stay"
    return max(
        options,
        key=lambda item: (
            min((abs(item[1][0] - d[0]) + abs(item[1][1] - d[1]) for d in defenders), default=view.rows + view.cols),
            -target_distances.get(item[1], view.rows * view.cols + 1),
        ),
    )[0]


def attacker_belief_action(belief: EvaderBelief) -> Action:
    """Replan from one attacker's sensor-limited belief, without predicted moves."""
    if belief.position is None:
        raise ValueError("CTF attacker perception is missing its position.")
    if not belief.known_goals and not belief.carrying_flag:
        return explore_action(belief, belief.position, belief.visible_pursuers)
    return attacker_action(
        attacker_view(belief), belief.position, belief.visible_pursuers, belief.carrying_flag,
        previous_position=belief.previous_position,
    )


def _unique_targets(
    options: dict[int, list[Position]], reserved: set[Position]
) -> dict[int, Position]:
    owners: dict[Position, int] = {}

    def assign(index: int, seen: set[Position], allow_reserved: bool) -> bool:
        for cell in options[index]:
            if cell in seen or (not allow_reserved and cell in reserved):
                continue
            seen.add(cell)
            owner = owners.get(cell)
            if owner is None or assign(owner, seen, allow_reserved):
                owners[cell] = index
                return True
        return False

    indices = sorted(options, key=lambda index: (len(options[index]), index))
    for index in indices:
        assign(index, set(), False)
    for index in indices:
        if index not in owners.values():
            assign(index, set(), True)
    return {index: cell for cell, index in owners.items()}


def assign_defender_targets(
    view: Scenario,
    defender_positions: tuple[Position, ...],
    attacker_estimate: Position | None,
    flag_state: FlagState,
    config: CTFConfig = CTFConfig(),
) -> list[Position]:
    """Assign guard and interception targets independent of the chosen planner.

    A deterministic augmenting-path match prefers the earliest reachable
    interception cells while maximizing distinct interceptor targets.
    """
    if not view.goals or not view.bases:
        raise ValueError("CTF strategy requires a flag home and at least one base.")
    if config.guards < 0 or config.guard_radius < 0:
        raise ValueError("Guard count and radius must be non-negative.")
    flag = view.goals[0]
    defender_distances = [_routes(view, position, view.pursuer_speed)[0] for position in defender_positions]
    flag_ring = _routes(view, flag, 1)[0]
    approach = attacker_estimate if attacker_estimate is not None else view.bases[0]
    approach_distances = _routes(view, approach, 1)[0]
    guard_indices = (
        [] if flag_state.carrier is not None
        else _guard_indices_from_distances(defender_distances, flag, config.guards)
    )
    assignments: list[Position | None] = [None] * len(defender_positions)
    used: set[Position] = set()
    for index in guard_indices:
        candidates = [
            cell for cell, distance in flag_ring.items()
            if distance <= config.guard_radius and cell in defender_distances[index]
        ]
        unused = [cell for cell in candidates if cell not in used]
        candidates = unused or candidates
        target = min(
            candidates,
            key=lambda cell: (
                approach_distances.get(cell, float("inf")),
                defender_distances[index][cell],
                cell,
            ),
            default=defender_positions[index],
        )
        assignments[index] = target
        used.add(target)

    interceptors = [index for index in range(len(defender_positions)) if index not in guard_indices]
    if attacker_estimate is not None:
        attacker_distances, previous = _routes(view, attacker_estimate, view.evader_speed)
        destinations = view.bases if flag_state.carrier is not None else (flag,)
        reachable = [target for target in destinations if target in attacker_distances]
        if reachable:
            destination = min(reachable, key=lambda cell: (attacker_distances[cell], cell))
            route = _path(attacker_estimate, destination, previous)
        else:
            route = [attacker_estimate]
        options = {
            index: [
                cell for cell in route
                if defender_distances[index].get(cell, float("inf")) <= attacker_distances[cell]
            ]
            for index in interceptors
        }
        matches = _unique_targets(options, used)
        for index in interceptors:
            assignments[index] = matches.get(
                index, options[index][0] if options[index] else attacker_estimate
            )
            if assignments[index] == defender_positions[index] and attacker_estimate != defender_positions[index]:
                assignments[index] = attacker_estimate
    else:
        flag_distances, previous = _routes(view, flag, 1)
        reachable_bases = [base for base in view.bases if base in flag_distances]
        base = min(reachable_bases, key=lambda cell: (flag_distances[cell], cell)) if reachable_bases else flag
        route = _path(flag, base, previous)
        options = {}
        for order, index in enumerate(interceptors):
            desired = (order + 1) * (len(route) - 1) / (len(interceptors) + 1)
            options[index] = sorted(
                (cell for cell in route if cell in defender_distances[index]),
                key=lambda cell: (abs(route.index(cell) - desired), defender_distances[index][cell]),
            )
        matches = _unique_targets(options, used)
        for index in interceptors:
            assignments[index] = matches.get(
                index, options[index][0] if options[index] else defender_positions[index]
            )
    return [target if target is not None else defender_positions[index] for index, target in enumerate(assignments)]


def _guard_indices_from_distances(
    distances: list[dict[Position, int]], flag: Position, guards: int
) -> list[int]:
    return sorted(
        range(len(distances)),
        key=lambda index: (distances[index].get(flag, float("inf")), index),
    )[:guards]


def guard_indices(view: Scenario, positions: tuple[Position, ...], config: CTFConfig) -> list[int]:
    """Return the same guards selected by assign_defender_targets."""
    if not view.goals:
        raise ValueError("CTF strategy requires a flag home.")
    distances = [_routes(view, position, view.pursuer_speed)[0] for position in positions]
    return _guard_indices_from_distances(distances, view.goals[0], config.guards)


def plan_for_targets(
    planner: Callable[[Scenario], dict],
    scenario_view: Scenario,
    assignments: list[Position],
) -> list[Action]:
    """Plan once per target group, preserving defender order across planners."""
    positions = scenario_view.pursuers
    if len(assignments) != len(positions):
        raise ValueError("Each defender must have exactly one target.")
    groups: dict[Position, list[int]] = {}
    for index, target in enumerate(assignments):
        groups.setdefault(target, []).append(index)
    actions: list[Action] = ["stay"] * len(positions)
    for target, indices in groups.items():
        group_view = replace(
            scenario_view,
            pursuers=tuple(positions[index] for index in indices),
            evaders=(target,),
        )
        paths = planner(group_view)["paths"]
        for group_index, index in enumerate(indices):
            position = positions[index]
            if position == target:
                continue
            path = paths[group_index] if group_index < len(paths) else []
            proposed = path[0] if path else None
            if proposed in ACTION_ORDER and (
                proposed == "stay" or _move(scenario_view, position, proposed, scenario_view.pursuer_speed) != position
            ):
                actions[index] = proposed
            else:
                _, previous = _routes(scenario_view, position, scenario_view.pursuer_speed)
                actions[index] = _first_action(position, target, previous)
    return actions
