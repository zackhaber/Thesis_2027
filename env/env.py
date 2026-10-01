from dataclasses import dataclass
from typing import Literal

from .game_modes import GameMode
from .level_generator import Position, Scenario

Action = Literal["stay", "up", "down", "left", "right"]
MOVES: dict[Action, tuple[int, int]] = {"stay": (0, 0), "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPPOSITE_ACTION: dict[Action, Action] = {"stay": "stay", "up": "down", "down": "up", "left": "right", "right": "left"}
GOAL_MODES = (GameMode.REACH_AVOID, GameMode.MAZE)

@dataclass
class State:
    pursuers: list[Position]
    evaders: list[Position]
    step_count: int

class PursuitEvasionEnv:
    def __init__(self, scenario: Scenario, mode: GameMode = GameMode.PURE_PURSUIT, max_steps: int = 200):
        self.scenario = scenario
        self.mode = mode
        self.max_steps = max_steps
        self.state = State([], [], 0)
        self.done = False

    def reset(self) -> dict:
        self.state = State(list(self.scenario.pursuers), list(self.scenario.evaders), 0)
        self.done = False
        return self.observe()

    def observe(self) -> dict:
        return {
            "pursuers": tuple(self.state.pursuers),
            "evaders": tuple(self.state.evaders),
            "barrels": self.scenario.barrels,
            "walls": self.scenario.walls,
            "goals": self.scenario.goals,
            "step": self.state.step_count,
        }

    def _wall_blocks(self, position: Position, action: Action) -> bool:
        if action == "stay":
            return False
        delta_row, delta_col = MOVES[action]
        neighbor = (position[0] + delta_row, position[1] + delta_col)
        wall_on_this_cell = (position, action) in self.scenario.walls
        wall_on_neighbor = (neighbor, OPPOSITE_ACTION[action]) in self.scenario.walls
        return wall_on_this_cell or wall_on_neighbor

    def _move(self, position: Position, action: Action, speed: int) -> Position:
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
        current_positions = list(self.state.pursuers)
        proposed_positions = [self._move(position, action, self.scenario.pursuer_speed) for position, action in zip(current_positions, actions)]
        proposal_counts = {position: proposed_positions.count(position) for position in proposed_positions}
        resolved_positions: list[Position] = []

        for current, proposed in zip(current_positions, proposed_positions):
            if proposal_counts[proposed] == 1 and proposed not in resolved_positions:
                resolved_positions.append(proposed)
                continue
            if current not in resolved_positions:
                resolved_positions.append(current)
                continue
            alternatives = [self._move(current, action, self.scenario.pursuer_speed) for action in MOVES]
            alternative = next((position for position in alternatives if position not in resolved_positions), current)
            resolved_positions.append(alternative)
        return resolved_positions

    def step(self, pursuer_actions: list[Action], evader_actions: list[Action]) -> tuple[dict, dict, bool, dict]:
        if self.done:
            raise RuntimeError("The episode is complete; call reset() before stepping again.")
        if len(pursuer_actions) != len(self.state.pursuers) or len(evader_actions) != len(self.state.evaders):
            raise ValueError("One action is required for every agent.")
        self.state.pursuers = self._move_pursuers(pursuer_actions)
        self.state.evaders = [self._move(pos, action, self.scenario.evader_speed) for pos, action in zip(self.state.evaders, evader_actions)]
        self.state.step_count += 1
        captured = any(evader in self.state.pursuers for evader in self.state.evaders)
        reached_goal = self.mode in GOAL_MODES and any(evader in self.scenario.goals for evader in self.state.evaders)
        timed_out = self.state.step_count >= self.max_steps
        self.done = captured or reached_goal or timed_out
        pursuer_reward = 1.0 if captured else (-0.01 if not self.done else 0.0)
        evader_reward = -1.0 if captured else (1.0 if reached_goal else -0.01)
        info = {"captured": captured, "reached_goal": reached_goal, "time": self.state.step_count}
        return self.observe(), {"pursuers": pursuer_reward, "evaders": evader_reward}, self.done, info
