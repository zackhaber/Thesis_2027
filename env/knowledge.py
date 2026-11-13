"""Partial-observability models for pursuit-evasion environments."""

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum

from .env import Action, MOVES, OPPOSITE_ACTION, PursuitEvasionEnv
from .game_modes import GameMode
from .scenario_generator import Position, Scenario


class KnowledgeMode(str, Enum):
    KNOWN = "Known"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class SensorModel:
    radius: int = 4
    line_of_sight: bool = True


@dataclass(frozen=True)
class CTFKnowledgeConfig:
    """CTF map and enemies are sensor-limited for both teams.

    Defenders know flag home and bases; attackers know bases and optionally
    flag home. A pickup alert reveals only that the flag was taken at home.
    Last sightings are usable while their age is less than last_seen_steps.
    """
    attacker_knows_flag: bool = True
    flag_taken_alert: bool = True
    last_seen_steps: int = 8

    def __post_init__(self) -> None:
        if self.last_seen_steps < 0:
            raise ValueError("last_seen_steps must be non-negative.")


@dataclass
class PursuerBelief:
    known_barrels: set[Position] = field(default_factory=set)
    known_walls: set[tuple[Position, str]] = field(default_factory=set)
    known_goals: set[Position] = field(default_factory=set)
    visible_evaders: set[Position] = field(default_factory=set)
    last_seen_evader: tuple[Position, int] | None = None
    explored_cells: set[Position] = field(default_factory=set)
    contaminated_cells: set[Position] = field(default_factory=set)
    cleared_cells: set[Position] = field(default_factory=set)
    pursuer_positions: tuple[Position, ...] = ()
    rows: int = 0
    cols: int = 0
    sensor: SensorModel = field(default_factory=SensorModel)
    ctf_template: Scenario | None = None
    flag_taken: bool = False
    flag_alert_position: Position | None = None
    flag_alert_step: int | None = None
    visible_flag_carrier: Position | None = None
    current_step: int = 0
    flag_alert_checked: bool = False

    def coverage(self) -> float:
        total = self.rows * self.cols
        return len(self.explored_cells) / total * 100 if total else 0.0


@dataclass
class EvaderBelief:
    known_barrels: set[Position] = field(default_factory=set)
    known_walls: set[tuple[Position, str]] = field(default_factory=set)
    known_goals: set[Position] = field(default_factory=set)
    visible_pursuers: set[Position] = field(default_factory=set)
    explored_cells: set[Position] = field(default_factory=set)
    rows: int = 0
    cols: int = 0
    sensor: SensorModel = field(default_factory=SensorModel)
    ctf_template: Scenario | None = None
    position: Position | None = None
    carrying_flag: bool = False
    previous_position: Position | None = None
    observed_step: int = -1


def _neighbor(position: Position, direction: str) -> Position:
    delta_row, delta_col = MOVES[direction]  # type: ignore[index]
    return position[0] + delta_row, position[1] + delta_col


def _wall_blocks(
    position: Position, direction: str, walls: set[tuple[Position, str]]
) -> bool:
    neighbor = _neighbor(position, direction)
    return (
        (position, direction) in walls
        or (neighbor, OPPOSITE_ACTION[direction]) in walls
    )


def _visible_cells(
    origin: Position,
    rows: int,
    cols: int,
    barrels: set[Position],
    walls: set[tuple[Position, str]],
    sensor: SensorModel,
) -> set[Position]:
    visible = {origin}
    pending = deque([(origin, 0)])
    while pending:
        position, distance = pending.popleft()
        if distance >= sensor.radius:
            continue
        for direction in ("up", "down", "left", "right"):
            candidate = _neighbor(position, direction)
            if not (0 <= candidate[0] < rows and 0 <= candidate[1] < cols):
                continue
            if _wall_blocks(position, direction, walls):
                continue
            if candidate in barrels:
                visible.add(candidate)
                continue
            if candidate in visible:
                continue
            visible.add(candidate)
            pending.append((candidate, distance + 1))
    return visible


def _ctf_visible_cells(
    origin: Position, rows: int, cols: int, barrels: set[Position],
    walls: set[tuple[Position, str]], sensor: SensorModel,
) -> set[Position]:
    """Trace grid lines; walls and barrels occlude cells beyond them."""
    visible = {origin}
    for row in range(max(0, origin[0] - sensor.radius), min(rows, origin[0] + sensor.radius + 1)):
        for col in range(max(0, origin[1] - sensor.radius), min(cols, origin[1] + sensor.radius + 1)):
            dr, dc = row - origin[0], col - origin[1]
            if abs(dr) + abs(dc) > sensor.radius or (row, col) == origin:
                continue
            if not sensor.line_of_sight:
                visible.add((row, col))
                continue
            steps = max(abs(dr), abs(dc))
            current = origin
            blocked = False
            for step in range(1, steps + 1):
                next_row = origin[0] + round(dr * step / steps)
                next_col = origin[1] + round(dc * step / steps)
                row_direction = "down" if next_row > current[0] else "up"
                col_direction = "right" if next_col > current[1] else "left"
                if next_row != current[0] and next_col != current[1]:
                    vertical = (next_row, current[1])
                    horizontal = (current[0], next_col)
                    if (
                        vertical in barrels or horizontal in barrels
                        or _wall_blocks(current, row_direction, walls)
                        or _wall_blocks(current, col_direction, walls)
                        or _wall_blocks(vertical, col_direction, walls)
                        or _wall_blocks(horizontal, row_direction, walls)
                    ):
                        blocked = True
                        break
                elif next_row != current[0] and _wall_blocks(current, row_direction, walls):
                    blocked = True
                    break
                elif next_col != current[1] and _wall_blocks(current, col_direction, walls):
                    blocked = True
                    break
                current = (next_row, next_col)
                if current in barrels and current != (row, col):
                    blocked = True
                    break
            if not blocked:
                visible.add((row, col))
    return visible


def _known_neighbors(
    position: Position,
    rows: int,
    cols: int,
    barrels: set[Position],
    walls: set[tuple[Position, str]],
):
    for direction in ("up", "down", "left", "right"):
        candidate = _neighbor(position, direction)
        if (
            0 <= candidate[0] < rows
            and 0 <= candidate[1] < cols
            and candidate not in barrels
            and not _wall_blocks(position, direction, walls)
        ):
            yield candidate, direction


def _update_contamination(belief: PursuerBelief, observed: set[Position]) -> None:
    all_cells = {
        (row, col)
        for row in range(belief.rows)
        for col in range(belief.cols)
    }
    belief.explored_cells.update(observed)
    belief.cleared_cells.update(observed)
    belief.cleared_cells.difference_update(belief.contaminated_cells)
    unobserved = all_cells - observed
    recontaminated = {
        cell
        for cell in unobserved
        if any(
            (cell[0] + row_delta, cell[1] + col_delta)
            in belief.contaminated_cells
            for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
    }
    belief.contaminated_cells.update(recontaminated)
    belief.cleared_cells.difference_update(recontaminated)


class Perception:
    """Maintains team-shared pursuer and evader beliefs."""

    def __init__(
        self,
        sensor: SensorModel | None = None,
        pursuer_belief: PursuerBelief | None = None,
        evader_belief: EvaderBelief | None = None,
        ctf_config: CTFKnowledgeConfig | None = None,
    ):
        self.sensor = sensor or SensorModel()
        self.pursuer_belief = pursuer_belief or PursuerBelief(sensor=self.sensor)
        self.evader_belief = evader_belief or EvaderBelief(sensor=self.sensor)
        self.ctf_config = ctf_config or CTFKnowledgeConfig()
        self.ctf_attacker_beliefs: list[EvaderBelief] = []

    def update(
        self, env: PursuitEvasionEnv, *, require_explored_evaders: bool = False
    ) -> None:
        """Update beliefs, optionally detecting evaders only in previously explored cells."""
        scenario = env.scenario
        state = env.state
        walls = set(scenario.walls)
        barrels = set(scenario.barrels)
        pursuer_visible = set()
        visible_cells = _ctf_visible_cells if env.mode == GameMode.CAPTURE_THE_FLAG else _visible_cells
        for position in state.pursuers:
            pursuer_visible.update(
                visible_cells(
                    position, scenario.rows, scenario.cols, barrels, walls, self.sensor
                )
            )
        self.pursuer_belief.rows = scenario.rows
        self.pursuer_belief.cols = scenario.cols
        self.pursuer_belief.pursuer_positions = tuple(state.pursuers)
        self.pursuer_belief.known_barrels.update(barrels.intersection(pursuer_visible))
        self.pursuer_belief.known_goals.update(set(scenario.goals).intersection(pursuer_visible))
        self.pursuer_belief.known_walls.update(
            edge for edge in walls if edge[0] in pursuer_visible
            or _neighbor(edge[0], edge[1]) in pursuer_visible
        )
        visible_evaders = set(state.evaders).intersection(pursuer_visible)
        if require_explored_evaders:
            visible_evaders.intersection_update(self.pursuer_belief.explored_cells)
        self.pursuer_belief.visible_evaders = visible_evaders
        if self.pursuer_belief.visible_evaders:
            self.pursuer_belief.last_seen_evader = (
                min(self.pursuer_belief.visible_evaders) if env.mode == GameMode.CAPTURE_THE_FLAG else next(iter(self.pursuer_belief.visible_evaders)),
                state.step_count,
            )
        elif (
            env.mode == GameMode.CAPTURE_THE_FLAG
            and self.pursuer_belief.last_seen_evader is not None
            and self.pursuer_belief.last_seen_evader[0] in pursuer_visible
        ):
            self.pursuer_belief.last_seen_evader = None
        _update_contamination(self.pursuer_belief, pursuer_visible)

        if env.mode == GameMode.CAPTURE_THE_FLAG:
            template = replace(
                scenario, pursuers=(), evaders=(), barrels=frozenset(),
                walls=frozenset(), goals=(), bases=(), seed=0,
            )
            defender = self.pursuer_belief
            defender.known_goals.add(scenario.goals[0])
            defender.ctf_template = replace(template, goals=(scenario.goals[0],), bases=scenario.bases)
            defender.current_step = state.step_count
            carrier = state.flag_carrier
            defender.visible_flag_carrier = (
                state.evaders[carrier]
                if carrier is not None and state.evaders[carrier] in pursuer_visible
                else None
            )
            if carrier is not None and state.evaders[carrier] in pursuer_visible:
                defender.flag_taken = True
            if self.ctf_config.flag_taken_alert and env._pickup_step is not None:
                defender.flag_taken = True
                defender.flag_alert_position = scenario.goals[0]
                defender.flag_alert_step = env._pickup_step
                if scenario.goals[0] in pursuer_visible and scenario.goals[0] not in defender.visible_evaders:
                    defender.flag_alert_checked = True
            while len(self.ctf_attacker_beliefs) < len(state.evaders):
                self.ctf_attacker_beliefs.append(EvaderBelief(sensor=self.sensor))
            del self.ctf_attacker_beliefs[len(state.evaders):]
            for index, position in enumerate(state.evaders):
                belief = self.ctf_attacker_beliefs[index]
                visible = _ctf_visible_cells(position, scenario.rows, scenario.cols, barrels, walls, self.sensor)
                belief.rows, belief.cols = scenario.rows, scenario.cols
                if belief.observed_step != state.step_count:
                    belief.previous_position = belief.position
                    belief.observed_step = state.step_count
                belief.position = position
                belief.carrying_flag = carrier == index
                belief.explored_cells.update(visible)
                belief.known_barrels.update(barrels.intersection(visible))
                belief.known_walls.update(
                    edge for edge in walls if edge[0] in visible
                    or _neighbor(edge[0], edge[1]) in visible
                )
                if self.ctf_config.attacker_knows_flag or scenario.goals[0] in visible or belief.carrying_flag:
                    belief.known_goals.add(scenario.goals[0])
                belief.visible_pursuers = set(state.pursuers).intersection(visible)
                belief.ctf_template = replace(template, bases=scenario.bases)
            return

        self.evader_belief.rows = scenario.rows
        self.evader_belief.cols = scenario.cols
        self.evader_belief.explored_cells.update(
            _visible_cells(
                state.evaders[0], scenario.rows, scenario.cols, barrels, walls, self.sensor
            )
            if state.evaders
            else set()
        )
        if state.evaders:
            visible = _visible_cells(
                state.evaders[0], scenario.rows, scenario.cols, barrels, walls, self.sensor
            )
            self.evader_belief.known_barrels.update(barrels.intersection(visible))
            self.evader_belief.known_goals.update(set(scenario.goals).intersection(visible))
            self.evader_belief.known_walls.update(
                edge for edge in walls if edge[0] in visible
                or _neighbor(edge[0], edge[1]) in visible
            )
            self.evader_belief.visible_pursuers = set(state.pursuers).intersection(visible)
        else:
            self.evader_belief.visible_pursuers.clear()


def planner_view(belief: PursuerBelief, env: PursuitEvasionEnv | None = None) -> Scenario:
    """Build the planner's partial, optimistic scenario from a pursuer belief."""
    evaders = tuple(sorted(belief.visible_evaders))
    if belief.ctf_template is None and not evaders and belief.last_seen_evader is not None:
        evaders = (belief.last_seen_evader[0],)
    if belief.ctf_template is not None:
        return replace(
            belief.ctf_template, pursuers=belief.pursuer_positions, evaders=evaders,
            barrels=frozenset(belief.known_barrels), walls=frozenset(belief.known_walls),
        )
    if env is None:
        raise ValueError("An environment is required for a non-CTF planner view.")
    return replace(
        env.scenario,
        pursuers=belief.pursuer_positions,
        barrels=frozenset(belief.known_barrels),
        walls=frozenset(belief.known_walls),
        goals=tuple(belief.known_goals),
        evaders=evaders,
    )


def attacker_view(belief: EvaderBelief) -> Scenario:
    """Construct an attacker view exclusively from that attacker's belief."""
    if belief.ctf_template is None or belief.position is None:
        raise ValueError("CTF attacker perception has not been updated.")
    return replace(
        belief.ctf_template, pursuers=tuple(sorted(belief.visible_pursuers)),
        evaders=(belief.position,), barrels=frozenset(belief.known_barrels),
        walls=frozenset(belief.known_walls), goals=tuple(sorted(belief.known_goals)),
    )


def explore_action(
    belief: PursuerBelief | EvaderBelief,
    position: Position,
    blocked: set[Position] | None = None,
) -> Action:
    """Move toward the nearest frontier using only the known map."""
    known_barrels = belief.known_barrels
    blocked = blocked or set()
    queue = deque([(position, None)])
    visited = {position}
    while queue:
        current, first_action = queue.popleft()
        if current not in belief.explored_cells:
            return first_action or "stay"
        for candidate, action in _known_neighbors(
            current, belief.rows, belief.cols, known_barrels, belief.known_walls
        ):
            if candidate in visited or candidate in blocked:
                continue
            visited.add(candidate)
            queue.append((candidate, action if first_action is None else first_action))
    return "stay"


def target_action(
    belief: EvaderBelief,
    position: Position,
    target: Position,
    blocked: set[Position] | None = None,
) -> Action:
    """Move toward a known target while treating visible pursuers as obstacles."""
    blocked = blocked or set()
    queue = deque([(position, None)])
    visited = {position}
    while queue:
        current, first_action = queue.popleft()
        if current == target:
            return first_action or "stay"
        for candidate, action in _known_neighbors(
            current, belief.rows, belief.cols, belief.known_barrels, belief.known_walls
        ):
            if candidate in visited or candidate in blocked:
                continue
            visited.add(candidate)
            queue.append((candidate, action if first_action is None else first_action))
    return "stay"
