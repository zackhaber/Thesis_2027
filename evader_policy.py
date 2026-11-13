"""Shared evader policies used by training and UI evaluation."""

from env.env import Action, PursuitEvasionEnv
from env.game_modes import GameMode
from strategies.ctf import attacker_action

Position = tuple[int, int]
MOVEMENT_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")


def _distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _nearest_target(position: Position, targets: tuple[Position, ...]) -> Position:
    return min(targets, key=lambda target: _distance(position, target))


def _position_after_action(
    environment: PursuitEvasionEnv,
    position: Position,
    action: Action,
    speed: int = 1,
) -> Position:
    return environment._move(position, action, speed)


def _legal_actions(environment: PursuitEvasionEnv, position: Position) -> list[Action]:
    return [
        action
        for action in MOVEMENT_ACTIONS
        if _position_after_action(environment, position, action) != position
    ]


def _first_step_to_target(
    environment: PursuitEvasionEnv,
    start: Position,
    target: Position,
    blocked_positions: set[Position],
) -> Action | None:
    queue: list[tuple[Position, Action | None]] = [(start, None)]
    visited = {start}
    while queue:
        position, first_action = queue.pop(0)
        if position == target:
            return first_action
        for action in MOVEMENT_ACTIONS:
            candidate = _position_after_action(environment, position, action)
            if candidate == position or candidate in visited or candidate in blocked_positions:
                continue
            visited.add(candidate)
            queue.append(
                (candidate, action if first_action is None else first_action)
            )
    return None


def evader_action(
    environment: PursuitEvasionEnv,
    evader: Position,
    pursuer_positions: set[Position],
    *,
    attacker_index: int | None = None,
) -> Action:
    """Compute the existing UI fleeing policy without changing its behavior."""
    if environment.mode == GameMode.CAPTURE_THE_FLAG:
        carrier = environment.state.flag_carrier
        if attacker_index is None and carrier is not None:
            matches = [index for index, position in enumerate(environment.state.evaders) if position == evader]
            if len(matches) != 1:
                raise ValueError("attacker_index is required when attackers share a position.")
            attacker_index = matches[0]
        if attacker_index is not None and (
            not 0 <= attacker_index < len(environment.state.evaders)
            or environment.state.evaders[attacker_index] != evader
        ):
            raise ValueError("attacker_index must identify the supplied attacker position.")
        return attacker_action(
            environment.scenario,
            evader,
            pursuer_positions,
            carrier is not None and carrier == attacker_index,
            environment.state.evaders[carrier] if carrier is not None else environment.scenario.goals[0],
            previous_position=(
                environment.state.previous_evaders[attacker_index]
                if attacker_index is not None and attacker_index < len(environment.state.previous_evaders)
                else None
            ),
        )
    goal_mode = environment.mode in (GameMode.REACH_AVOID, GameMode.MAZE)

    if goal_mode:
        nearest_goal = _nearest_target(evader, tuple(environment.scenario.goals))
        safe_action = _first_step_to_target(
            environment, evader, nearest_goal, pursuer_positions
        )
        if safe_action is not None:
            return safe_action

    legal = _legal_actions(environment, evader)
    if not legal:
        return MOVEMENT_ACTIONS[0]

    def score_action(action: Action) -> tuple[int, int]:
        new_pos = _position_after_action(environment, evader, action)
        safety = min(_distance(new_pos, pursuer) for pursuer in pursuer_positions)
        goal_bonus = (
            -_distance(
                new_pos,
                _nearest_target(evader, tuple(environment.scenario.goals)),
            )
            if goal_mode
            else 0
        )
        return safety, goal_bonus

    return max(legal, key=score_action)
