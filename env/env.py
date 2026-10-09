from dataclasses import dataclass
from random import Random
from typing import Literal

from .game_modes import GameMode
from .level_generator import Position, Scenario

Action = Literal["stay", "up", "down", "left", "right"]
ACTION_ORDER: tuple[Action, ...] = ("stay", "up", "down", "left", "right")
MOVES: dict[Action, tuple[int, int]] = {"stay": (0, 0), "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPPOSITE_ACTION: dict[Action, Action] = {"stay": "stay", "up": "down", "down": "up", "left": "right", "right": "left"}
PURSUER_ALTERNATIVE_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right", "stay")
GOAL_MODES = (GameMode.REACH_AVOID, GameMode.MAZE)
@dataclass
class State:
    pursuers: list[Position]
    evaders: list[Position]
    step_count: int

class PursuitEvasionEnv:
    def __init__(self, scenario: Scenario, mode: GameMode = GameMode.PURE_PURSUIT, max_steps: int = 200, seed: int | None = None):
        # Keep the immutable scenario separate from the mutable episode state.
        self.scenario = scenario
        self.mode = mode
        self.max_steps = max_steps
        self.seed = scenario.seed if seed is None else seed
        self.rng = Random(self.seed)
        self.state = State([], [], 0)
        self.terminated = False
        self.truncated = False

    def reset(self) -> dict:
        # Recreate agent positions so a new episode starts from the generated layout.
        self.state = State(list(self.scenario.pursuers), list(self.scenario.evaders), 0)
        self.rng = Random(self.seed)
        self.terminated = False
        self.truncated = False
        return self.observe()

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def observe(self) -> dict:
        # Expose immutable tuples for positions while retaining scenario objects.
        agent_observations = {
            "pursuers": tuple(self._agent_observation(position, "pursuer") for position in self.state.pursuers),
            "evaders": tuple(self._agent_observation(position, "evader") for position in self.state.evaders),
        }
        return {
            "pursuers": tuple(self.state.pursuers),
            "evaders": tuple(self.state.evaders),
            "barrels": self.scenario.barrels,
            "walls": self.scenario.walls,
            "goals": self.scenario.goals,
            "step": self.state.step_count,
            "observations": agent_observations,
            "global_state": self._global_state(),
            "action_masks": self._action_masks(),
        }

    def _global_state(self) -> tuple[int, ...]:
        # Encode a fixed-size state: pursuers, evaders, barrels, goals, then walls.
        state: list[int] = []
        for row in range(self.scenario.rows):
            for col in range(self.scenario.cols):
                position = (row, col)
                state.extend((
                    int(position in self.state.pursuers),
                    int(position in self.state.evaders),
                    int(position in self.scenario.barrels),
                    int(position in self.scenario.goals),
                ))
                state.extend(int((position, direction) in self.scenario.walls) for direction in ("up", "down", "left", "right"))
        return tuple(state)

    def _valid_actions(self, position: Position, speed: int) -> tuple[bool, ...]:
        return tuple(action == "stay" or self._move(position, action, speed) != position for action in ACTION_ORDER)

    def _action_masks(self) -> dict[str, tuple[tuple[bool, ...], ...]]:
        return {
            "pursuers": tuple(self._valid_actions(position, self.scenario.pursuer_speed) for position in self.state.pursuers),
            "evaders": tuple(self._valid_actions(position, self.scenario.evader_speed) for position in self.state.evaders),
        }

    def _agent_observation(self, position: Position, role: str) -> dict:
        speed = self.scenario.pursuer_speed if role == "pursuer" else self.scenario.evader_speed
        return {
            "role": role,
            "position": position,
            "pursuers": tuple(self.state.pursuers),
            "evaders": tuple(self.state.evaders),
            "barrels": self.scenario.barrels,
            "walls": self.scenario.walls,
            "goals": self.scenario.goals,
            "step": self.state.step_count,
            "action_mask": self._valid_actions(position, speed),
        }

    def _wall_blocks(self, position: Position, action: Action) -> bool:
        # Check both sides of an edge because walls are stored as directed entries.
        if action == "stay":
            return False
        delta_row, delta_col = MOVES[action]
        neighbor = (position[0] + delta_row, position[1] + delta_col)
        wall_on_this_cell = (position, action) in self.scenario.walls
        wall_on_neighbor = (neighbor, OPPOSITE_ACTION[action]) in self.scenario.walls
        return wall_on_this_cell or wall_on_neighbor

    def _move(self, position: Position, action: Action, speed: int) -> Position:
        # Apply one action repeatedly, stopping at the first invalid cell or edge.
        row, col = position
        delta_row, delta_col = MOVES[action]
        for _ in range(speed):
            candidate = (row + delta_row, col + delta_col)
            inside_grid = 0 <= candidate[0] < self.scenario.rows and 0 <= candidate[1] < self.scenario.cols
            if not inside_grid:
                break
            if self._wall_blocks((row, col), action):
                break
            if candidate in self.scenario.barrels:
                break
            row, col = candidate
        return row, col

    def _move_pursuers(self, actions: list[Action]) -> list[Position]:
        # Resolve simultaneous destinations before committing the pursuer positions.
        current_positions = list(self.state.pursuers)
        proposed_positions = [self._move(position, action, self.scenario.pursuer_speed) for position, action in zip(current_positions, actions)]
        resolved_positions: list[Position | None] = [None] * len(current_positions)
        occupied: set[Position] = set()

        for proposed in dict.fromkeys(proposed_positions):
            # A random contender wins a shared destination instead of list order deciding.
            contenders = [index for index, position in enumerate(proposed_positions) if position == proposed]
            winner = contenders[0] if len(contenders) == 1 else self.rng.choice(contenders)
            if proposed not in occupied:
                resolved_positions[winner] = proposed
                occupied.add(proposed)

        for index, current in enumerate(current_positions):
            if resolved_positions[index] is not None:
                continue
            # Losers try directional alternatives before using stay as a last resort.
            alternatives = [self._move(current, action, self.scenario.pursuer_speed) for action in PURSUER_ALTERNATIVE_ACTIONS]
            alternative = next((position for position in alternatives if position not in occupied), current)
            resolved_positions[index] = alternative
            occupied.add(alternative)
        return [position for position in resolved_positions if position is not None]

    def step(self, pursuer_actions: list[Action], evader_actions: list[Action]) -> tuple[dict, dict, bool, bool, dict]:
        # Advance pursuers first, then block direct pursuer-evader endpoint swaps.
        if self.done:
            raise RuntimeError("The episode is complete; call reset() before stepping again.")
        if len(pursuer_actions) != len(self.state.pursuers) or len(evader_actions) != len(self.state.evaders):
            raise ValueError("One action is required for every agent.")
        previous_pursuers = list(self.state.pursuers)
        previous_evaders = list(self.state.evaders)
        moved_pursuers = self._move_pursuers(pursuer_actions)
        proposed_evaders = [self._move(pos, action, self.scenario.evader_speed) for pos, action in zip(previous_evaders, evader_actions)]
        moved_evaders = [
            # A swapping evader stays put so the pursuer captures it normally.
            previous_evader
            if any(
                previous_pursuer == proposed_evader and moved_pursuer == previous_evader
                for previous_pursuer, moved_pursuer in zip(previous_pursuers, moved_pursuers)
            )
            else proposed_evader
            for previous_evader, proposed_evader in zip(previous_evaders, proposed_evaders)
        ]
        self.state.pursuers = moved_pursuers
        self.state.evaders = moved_evaders
        self.state.step_count += 1
        # Evaluate terminal conditions only after all positions for this step settle.
        captured = any(evader in self.state.pursuers for evader in self.state.evaders)
        reached_goal = self.mode in GOAL_MODES and any(evader in self.scenario.goals for evader in self.state.evaders)
        timed_out = self.state.step_count >= self.max_steps
        self.terminated = captured or reached_goal
        self.truncated = timed_out and not self.terminated
        # Survival is the evader's objective, including surviving until the horizon.
        pursuer_reward = 1.0 if captured else (-0.01 if not self.done else 0.0)
        evader_reward = -1.0 if captured else (1.0 if reached_goal or self.truncated else -0.01)
        info = {
            "captured": captured,
            "reached_goal": reached_goal,
            "time": self.state.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        return self.observe(), {"pursuers": pursuer_reward, "evaders": evader_reward}, self.terminated, self.truncated, info
