"""Partial-observability models for pursuit-evasion environments."""

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum

from .env import Action, MOVES, OPPOSITE_ACTION, PursuitEvasionEnv
from .scenario_generator import Position, Scenario


class KnowledgeMode(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SensorModel:
    radius: int = 4
    line_of_sight: bool = True


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
    ):
        self.sensor = sensor or SensorModel()
        self.pursuer_belief = pursuer_belief or PursuerBelief(sensor=self.sensor)
        self.evader_belief = evader_belief or EvaderBelief(sensor=self.sensor)

    def update(self, env: PursuitEvasionEnv) -> None:
        """Read ground truth once and update both belief models."""
        scenario = env.scenario
        state = env.state
        walls = set(scenario.walls)
        barrels = set(scenario.barrels)
        pursuer_visible = set()
        for position in state.pursuers:
            pursuer_visible.update(
                _visible_cells(
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
        self.pursuer_belief.visible_evaders = set(state.evaders).intersection(pursuer_visible)
        if self.pursuer_belief.visible_evaders:
            self.pursuer_belief.last_seen_evader = (
                next(iter(self.pursuer_belief.visible_evaders)),
                state.step_count,
            )
        _update_contamination(self.pursuer_belief, pursuer_visible)

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


def planner_view(belief: PursuerBelief, env: PursuitEvasionEnv) -> Scenario:
    """Build the planner's partial, optimistic scenario from a pursuer belief."""
    evaders = tuple(belief.visible_evaders)
    if not evaders and belief.last_seen_evader is not None:
        evaders = (belief.last_seen_evader[0],)
    return replace(
        env.scenario,
        pursuers=belief.pursuer_positions,
        barrels=frozenset(belief.known_barrels),
        walls=frozenset(belief.known_walls),
        goals=tuple(belief.known_goals),
        evaders=evaders,
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
