"""
Implements the PursuitEvasionEnv grid environment.

Handles:
1. Multi-agent state tracking (pursuer/evader positions, step limits).
2. Grid physics (directional walls, impassable barrels, board boundaries).
3. Simultaneous movement conflict resolution between pursuers.
4. Observation generation (local agent masks & global state vector for MARL value mixers).
5. Reward calculation and episode termination/truncation checks.
"""

from dataclasses import dataclass
from random import Random
from typing import Literal
from .game_modes import GameMode
from .scenario_generator import Position, Scenario

# Define discrete movement actions
Action = Literal["stay", "up", "down", "left", "right"]
ACTION_ORDER: tuple[Action, ...] = ("stay", "up", "down", "left", "right")

# Grid coordinate deltas: (delta_row, delta_col)
MOVES: dict[Action, tuple[int, int]] = {
    "stay": (0, 0),
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Opposite actions used to verify walls attached to neighbor cells
OPPOSITE_ACTION: dict[Action, Action] = {
    "stay": "stay",
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

PURSUER_ALTERNATIVE_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right", "stay")
GOAL_MODES = (GameMode.REACH_AVOID, GameMode.MAZE)


@dataclass
class State:
    """Mutable state tracking current agent positions and episode step count."""
    pursuers: list[Position]
    evaders: list[Position]
    step_count: int


class PursuitEvasionEnv:
    """
    The main multi-agent pursuit-evasion grid world environment.
    """

    def __init__(
        self,
        scenario: Scenario,
        mode: GameMode = GameMode.PURE_PURSUIT,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        # Keep immutable scenario parameters separate from mutable episode state
        self.scenario = scenario
        self.mode = mode
        self.max_steps = max_steps
        self.seed = scenario.seed if seed is None else seed
        self.rng = Random(self.seed)

        # Dynamic state & episode flags
        self.state = State(pursuers=[], evaders=[], step_count=0)
        self.terminated = False
        self.truncated = False

    def reset(self) -> dict:
        """Resets the environment to start a fresh episode from the scenario layout."""
        self.state = State(
            pursuers=list(self.scenario.pursuers),
            evaders=list(self.scenario.evaders),
            step_count=0,
        )
        self.rng = Random(self.seed)
        self.terminated = False
        self.truncated = False
        return self.observe()

    @property
    def done(self) -> bool:
        """Episode ends if terminated (goal/capture) or truncated (timeout)."""
        return self.terminated or self.truncated

    def observe(self) -> dict:
        """
        Builds and returns the complete environment observation dictionary:
        - Positions of pursuers, evaders, goals, barrels, and walls.
        - Individual per-agent observations and valid action masks.
        - Fixed-size 1D global state vector for value mixers (e.g., QMIX).
        """
        agent_observations = {
            "pursuers": tuple(
                self._agent_observation(pos, "pursuer") for pos in self.state.pursuers
            ),
            "evaders": tuple(
                self._agent_observation(pos, "evader") for pos in self.state.evaders
            ),
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
        """
        Encodes the entire grid into a flat 1D binary feature vector for MARL value mixers.
        For each cell: [is_pursuer, is_evader, is_barrel, is_goal, wall_up, wall_down, wall_left, wall_right]
        """
        state_vector: list[int] = []
        for row in range(self.scenario.rows):
            for col in range(self.scenario.cols):
                pos = (row, col)

                # 1. Entity presence indicators
                state_vector.append(int(pos in self.state.pursuers))
                state_vector.append(int(pos in self.state.evaders))
                state_vector.append(int(pos in self.scenario.barrels))
                state_vector.append(int(pos in self.scenario.goals))

                # 2. Wall presence indicators in 4 directions
                for direction in ("up", "down", "left", "right"):
                    has_wall = (pos, direction) in self.scenario.walls
                    state_vector.append(int(has_wall))

        return tuple(state_vector)

    def _valid_actions(self, position: Position, speed: int) -> tuple[bool, ...]:
        """Returns a tuple of booleans indicating valid actions for an agent at `position`."""
        valid_list = []
        for action in ACTION_ORDER:
            if action == "stay":
                valid_list.append(True)
            else:
                # Valid if taking the action changes position (not blocked by wall/barrel/edge)
                new_position = self._move(position, action, speed)
                valid_list.append(new_position != position)
        return tuple(valid_list)

    def _action_masks(self) -> dict[str, tuple[tuple[bool, ...], ...]]:
        """Returns action validity masks for all pursuers and evaders."""
        pursuer_masks = tuple(
            self._valid_actions(pos, self.scenario.pursuer_speed)
            for pos in self.state.pursuers
        )
        evader_masks = tuple(
            self._valid_actions(pos, self.scenario.evader_speed)
            for pos in self.state.evaders
        )
        return {"pursuers": pursuer_masks, "evaders": evader_masks}

    def _agent_observation(self, position: Position, role: str) -> dict:
        """Constructs a local observation dictionary for a single agent."""
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
        """Checks if a directional wall blocks movement from `position` in `action` direction."""
        if action == "stay":
            return False

        delta_row, delta_col = MOVES[action]
        neighbor = (position[0] + delta_row, position[1] + delta_col)

        # Check for wall leaving current cell OR wall entering from neighbor cell
        wall_on_this_cell = (position, action) in self.scenario.walls
        wall_on_neighbor = (neighbor, OPPOSITE_ACTION[action]) in self.scenario.walls

        return wall_on_this_cell or wall_on_neighbor

    def _move(self, position: Position, action: Action, speed: int) -> Position:
        """
        Calculates the resulting position after moving `speed` steps in `action` direction.
        Stops early if blocked by board boundary, directional wall, or barrel.
        """
        row, col = position
        delta_row, delta_col = MOVES[action]

        for _ in range(speed):
            candidate = (row + delta_row, col + delta_col)

            # 1. Grid boundary check
            inside_grid = 0 <= candidate[0] < self.scenario.rows and 0 <= candidate[1] < self.scenario.cols
            if not inside_grid:
                break

            # 2. Directional wall check
            if self._wall_blocks((row, col), action):
                break

            # 3. Barrel obstacle check
            if candidate in self.scenario.barrels:
                break

            row, col = candidate

        return row, col

    def _move_pursuers(self, actions: list[Action]) -> list[Position]:
        """
        Simultaneously moves pursuers and resolves collisions when multiple 
        pursuers attempt to occupy the same cell.
        """
        current_positions = list(self.state.pursuers)
        proposed_positions = [
            self._move(pos, act, self.scenario.pursuer_speed)
            for pos, act in zip(current_positions, actions)
        ]

        resolved_positions: list[Position | None] = [None] * len(current_positions)
        occupied: set[Position] = set()

        # Step 1: Resolve contention for proposed cells.
        # If multiple pursuers propose the same destination, pick one winner at random.
        unique_proposed = dict.fromkeys(proposed_positions)
        for proposed in unique_proposed:
            contenders = [i for i, pos in enumerate(proposed_positions) if pos == proposed]
            winner = contenders[0] if len(contenders) == 1 else self.rng.choice(contenders)

            if proposed not in occupied:
                resolved_positions[winner] = proposed
                occupied.add(proposed)

        # Step 2: For pursuers whose proposed moves were rejected, try alternative directional moves
        for index, current_pos in enumerate(current_positions):
            if resolved_positions[index] is not None:
                continue  # Already successfully assigned a move

            alternatives = [
                self._move(current_pos, act, self.scenario.pursuer_speed)
                for act in PURSUER_ALTERNATIVE_ACTIONS
            ]
            
            # Pick the first unoccupied alternative cell (or stay in current position)
            fallback_pos = next(
                (pos for pos in alternatives if pos not in occupied),
                current_pos
            )
            resolved_positions[index] = fallback_pos
            occupied.add(fallback_pos)

        return [pos for pos in resolved_positions if pos is not None]

    def step(
        self, pursuer_actions: list[Action], evader_actions: list[Action]
    ) -> tuple[dict, dict, bool, bool, dict]:
        """
        Advances the environment by one step:
        1. Moves pursuers (resolving simultaneous destination conflicts).
        2. Moves evaders (preventing direct position swaps with pursuers).
        3. Evaluates termination (capture/goal) and truncation (step limit).
        4. Calculates team rewards.
        """
        if self.done:
            raise RuntimeError("Episode is complete; call reset() before stepping again.")

        if len(pursuer_actions) != len(self.state.pursuers) or len(evader_actions) != len(self.state.evaders):
            raise ValueError("Exactly one action is required for every agent.")

        previous_pursuers = list(self.state.pursuers)
        previous_evaders = list(self.state.evaders)

        # Step 1: Move pursuers
        moved_pursuers = self._move_pursuers(pursuer_actions)

        # Step 2: Propose evader moves
        proposed_evaders = [
            self._move(pos, act, self.scenario.evader_speed)
            for pos, act in zip(previous_evaders, evader_actions)
        ]

        # Step 3: Prevent direct endpoint swapping (e.g., pursuer & evader swapping cells in 1 turn).
        # If a swap is detected, force the evader to stay put so capture happens naturally.
        moved_evaders = []
        for prev_evader, prop_evader in zip(previous_evaders, proposed_evaders):
            is_swapping = any(
                prev_pursuer == prop_evader and moved_pursuer == prev_evader
                for prev_pursuer, moved_pursuer in zip(previous_pursuers, moved_pursuers)
            )
            if is_swapping:
                moved_evaders.append(prev_evader)
            else:
                moved_evaders.append(prop_evader)

        # Update environment state
        self.state.pursuers = moved_pursuers
        self.state.evaders = moved_evaders
        self.state.step_count += 1

        # Step 4: Evaluate terminal conditions
        captured = any(evader in self.state.pursuers for evader in self.state.evaders)
        reached_goal = (
            self.mode in GOAL_MODES
            and any(evader in self.scenario.goals for evader in self.state.evaders)
        )
        timed_out = self.state.step_count >= self.max_steps

        self.terminated = captured or reached_goal
        self.truncated = timed_out and not self.terminated

        # Step 5: Reward calculation
        pursuer_reward = 1.0 if captured else (-0.01 if not self.done else 0.0)
        evader_reward = -1.0 if captured else (1.0 if (reached_goal or self.truncated) else -0.01)

        info = {
            "captured": captured,
            "reached_goal": reached_goal,
            "time": self.state.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

        return (
            self.observe(),
            {"pursuers": pursuer_reward, "evaders": evader_reward},
            self.terminated,
            self.truncated,
            info,
        )