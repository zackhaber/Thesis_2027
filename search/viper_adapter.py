"""Inference-only adapter from ViPER graph actions to project Actions."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Callable

from env.env import ACTION_ORDER, MOVES, OPPOSITE_ACTION, Action
from env.knowledge import PursuerBelief, explore_action, planner_view
from env.scenario_generator import Scenario

LOGGER = logging.getLogger(__name__)
SUPPORTED_AGENT_COUNT = 4
DEFAULT_UNSEEN_STEPS = 5

PlannerCallback = Callable[
    [PursuerBelief, tuple[tuple[int, int], ...], str], list[Action]
]


class ViperSearcher:
    """Runs the pretrained ViPER search policy and hands off to pursuit planners."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        *,
        selected_algorithm: str = "CBS",
        switch_on_detection: bool = True,
        unseen_steps: int = DEFAULT_UNSEEN_STEPS,
        planner_callback: PlannerCallback | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "ViPER"
            / "model"
            / "viper"
            / "checkpoint.pth"
        )
        self.selected_algorithm = selected_algorithm
        self.switch_on_detection = switch_on_detection
        self.unseen_steps = max(1, unseen_steps)
        self.planner_callback = planner_callback
        self._policy = None
        self._load_attempted = False
        self._fallback_logged = False
        self._handoff_active = False
        self._unseen_count = 0
        self._environment = None
        self.fallback_count = 0
        self.is_active = False
        self._dependency_error: Exception | None = None
        self.sensor_radius_cells = self._verified_sensor_radius()

    @staticmethod
    def _verified_sensor_radius() -> int:
        """Derive cells from the verified ViPER meter constants."""
        try:
            from third_party.viper_inference.constants import NODE_RESOLUTION, SENSOR_RANGE
        except ImportError:
            return 5
        return int(round(SENSOR_RANGE / NODE_RESOLUTION))

    def set_environment(self, environment) -> None:
        """Provide the live environment for planner_view during handoff."""
        self._environment = environment

    def reset_episode(self) -> None:
        self._handoff_active = False
        self._unseen_count = 0

    def resolve_conflicts(self, environment, actions: list[Action]):
        """Delegate conflict resolution to the environment's canonical method."""
        return environment._move_pursuers(actions)

    def _log_fallback(self, reason: Exception | str) -> None:
        if not self._fallback_logged:
            LOGGER.warning("ViPER search unavailable; using explore_action fallback: %s", reason)
            self._fallback_logged = True

    def _load_policy(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from third_party.viper_inference.inference import ViperPolicy

            self._policy = ViperPolicy(self.checkpoint_path)
            self.is_active = True
        except (ImportError, ModuleNotFoundError, FileNotFoundError, OSError, KeyError, RuntimeError) as error:
            self._dependency_error = error
            self._log_fallback(error)

    @staticmethod
    def _belief_scenario(
        belief: PursuerBelief,
        pursuer_positions: tuple[tuple[int, int], ...],
    ) -> Scenario:
        evaders = tuple(belief.visible_evaders)
        if not evaders and belief.last_seen_evader is not None:
            evaders = (belief.last_seen_evader[0],)
        return Scenario(
            rows=belief.rows,
            cols=belief.cols,
            pursuers=pursuer_positions,
            evaders=evaders,
            barrels=frozenset(belief.known_barrels),
            walls=frozenset(belief.known_walls),
            goals=tuple(belief.known_goals),
            pursuer_speed=1,
            evader_speed=1,
            seed=0,
        )

    def _planner_actions(
        self,
        belief: PursuerBelief,
        pursuer_positions: tuple[tuple[int, int], ...],
    ) -> list[Action]:
        if self.planner_callback is not None:
            proposed = self.planner_callback(belief, pursuer_positions, self.selected_algorithm)
            if len(proposed) != len(pursuer_positions):
                raise ValueError("Pursuit planner returned one action per pursuer incorrectly")
            return [
                action
                if action in ACTION_ORDER and self._action_is_known_legal(belief, position, action)
                else explore_action(belief, position)
                for position, action in zip(pursuer_positions, proposed)
            ]

        scenario = (
            planner_view(belief, self._environment)
            if self._environment is not None
            else self._belief_scenario(belief, pursuer_positions)
        )
        from planners.cbs import cbs_plan
        from planners.mstar import mstar_plan
        from planners.voronoi import voronoi_plan

        planners = {"CBS": cbs_plan, "M*": mstar_plan, "Voronoi": voronoi_plan}
        if self.selected_algorithm not in planners:
            raise ValueError(f"Unsupported pursuit planner: {self.selected_algorithm}")
        result = planners[self.selected_algorithm](scenario)
        actions: list[Action] = []
        for index, position in enumerate(pursuer_positions):
            paths = result.get("paths", [])
            path = paths[index] if index < len(paths) else []
            candidate = path[0] if path else None
            if candidate in ACTION_ORDER and self._action_is_known_legal(belief, position, candidate):
                actions.append(candidate)
            else:
                actions.append(explore_action(belief, position))
        return actions

    def _sync_sensor_radius(self, belief: PursuerBelief) -> None:
        if belief.sensor.radius != self.sensor_radius_cells:
            belief.sensor = replace(belief.sensor, radius=self.sensor_radius_cells)

    @property
    def status_text(self) -> str:
        return "Search: ViPER" if self.is_active else f"Search: Fallback ({self.fallback_count})"

    @property
    def handoff_active(self) -> bool:
        return self._handoff_active

    @staticmethod
    def _known_free_cells(belief: PursuerBelief) -> set[tuple[int, int]]:
        return (
            (set(belief.explored_cells) | set(belief.cleared_cells))
            - set(belief.known_barrels)
            - set(belief.contaminated_cells)
        )

    def _rasterize_belief(self, belief: PursuerBelief, positions):
        from third_party.viper_inference.constants import FREE, PIXELS_PER_CELL, UNKNOWN
        from third_party.viper_inference.raster import rasterize

        scenario = self._belief_scenario(belief, tuple(positions))
        raster = rasterize(scenario)
        known_free = self._known_free_cells(belief)
        for row in range(belief.rows):
            for col in range(belief.cols):
                if (row, col) not in known_free and (row, col) not in belief.known_barrels:
                    start_row, end_row = row * PIXELS_PER_CELL, (row + 1) * PIXELS_PER_CELL
                    start_col, end_col = col * PIXELS_PER_CELL, (col + 1) * PIXELS_PER_CELL
                    raster.image[start_row:end_row, start_col:end_col] = UNKNOWN
        for row, col in belief.cleared_cells:
            if (row, col) not in belief.known_barrels:
                raster.image[
                    row * PIXELS_PER_CELL:(row + 1) * PIXELS_PER_CELL,
                    col * PIXELS_PER_CELL:(col + 1) * PIXELS_PER_CELL,
                ] = FREE
        return raster, known_free

    @staticmethod
    def _action_is_known_legal(belief: PursuerBelief, position, action: Action) -> bool:
        if action == "stay":
            return True
        delta_row, delta_col = MOVES[action]
        candidate = position[0] + delta_row, position[1] + delta_col
        if not (0 <= candidate[0] < belief.rows and 0 <= candidate[1] < belief.cols):
            return False
        if candidate in belief.known_barrels:
            return False
        return (
            (position, action) not in belief.known_walls
            and (candidate, OPPOSITE_ACTION[action]) not in belief.known_walls
        )

    @classmethod
    def _first_step_to_target(
        cls,
        belief: PursuerBelief,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> Action | None:
        """BFS equivalent of ui._first_step_to_target on the known map."""
        queue: deque[tuple[tuple[int, int], Action | None]] = deque([(start, None)])
        visited = {start}
        known_free = cls._known_free_cells(belief)
        if target not in known_free and target != start:
            return None
        while queue:
            position, first_action = queue.popleft()
            if position == target:
                return first_action
            for action in ("stay", "up", "down", "left", "right"):
                if action == "stay" or not cls._action_is_known_legal(belief, position, action):
                    continue
                delta_row, delta_col = MOVES[action]
                candidate = position[0] + delta_row, position[1] + delta_col
                if candidate in visited or candidate not in known_free:
                    continue
                visited.add(candidate)
                queue.append((candidate, action if first_action is None else first_action))
        return None

    def _search_actions(self, belief: PursuerBelief, positions):
        try:
            raster, known_free = self._rasterize_belief(belief, positions)
        except ImportError as error:
            self._dependency_error = error
            self._log_fallback(error)
            self.is_active = False
            self.fallback_count += len(positions)
            return [explore_action(belief, position) for position in positions]
        del raster  # The graph builder consumes the same native-scale belief below.
        if not self.is_active:
            self.fallback_count += len(positions)
            return [explore_action(belief, position) for position in positions]
        if len(positions) != SUPPORTED_AGENT_COUNT:
            raise ValueError(
                f"ViPER checkpoint supports exactly {SUPPORTED_AGENT_COUNT} pursuers; "
                f"received {len(positions)}"
            )
        try:
            choices = self._policy.choose_many(
                positions,
                known_free,
                belief.rows,
                belief.cols,
                set(belief.known_barrels),
                set(belief.known_walls),
                positions,
            )
        except ValueError:
            raise
        actions: list[Action] = []
        for position, (target, _) in zip(positions, choices):
            action = self._first_step_to_target(belief, position, tuple(target))
            if action is None:
                self.fallback_count += 1
                action = explore_action(belief, position)
            actions.append(action)
        return actions

    def act(self, belief: PursuerBelief, pursuer_positions) -> list[Action]:
        """Return one search or pursuit action for each pursuer."""
        positions = tuple(pursuer_positions)
        if len(positions) != SUPPORTED_AGENT_COUNT and self._load_attempted and self.is_active:
            raise ValueError(
                f"ViPER checkpoint supports exactly {SUPPORTED_AGENT_COUNT} pursuers; "
                f"received {len(positions)}"
            )
        self._sync_sensor_radius(belief)
        belief.pursuer_positions = positions
        self._load_policy()
        if not self.switch_on_detection:
            self._handoff_active = False
            self._unseen_count = 0
        elif belief.visible_evaders:
            self._handoff_active = True
            self._unseen_count = 0
        elif self._handoff_active:
            self._unseen_count += 1
            if self._unseen_count >= self.unseen_steps:
                self._handoff_active = False
        if self._handoff_active:
            return self._planner_actions(belief, positions)
        return self._search_actions(belief, positions)
