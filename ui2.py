"""
Pursuit-Evasion GUI (simplified and documented version)

This file has two main parts:

    PART 1 - The "brain": plain functions that decide where each pursuer
             and evader should move on every timestep.

    PART 2 - The "window": a Tkinter app (PursuitEvasionApp) that shows the
             menus, draws the board, and runs the simulation on a timer.

The behaviour is identical to the original file. Only the layout of the
code changed: one step per line, helper functions for repeated work, and
comments explaining the reasoning.
"""

import tkinter as tk
from tkinter import ttk
from collections import deque
from dataclasses import replace
from pathlib import Path
from random import SystemRandom
from threading import Thread
from time import perf_counter
from typing import cast

import customtkinter as ctk

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.env import Action
from env.game_modes import GameMode
from planners import cbs_plan, mstar_plan, voronoi_plan

# A position on the board is always (row, col).
Position = tuple[int, int]

# The four moves an agent is allowed to try.
MOVEMENT_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")

# Human-readable names for each game mode (used in menus and the status panel).
MODE_LABELS = {
    GameMode.PURE_PURSUIT: "Pure Pursuit",
    GameMode.REACH_AVOID: "Reach and Avoid",
    GameMode.MAZE: "Maze",
}

# Algorithm names shown on the algorithm menu.
CLASSICAL_ALGORITHMS = ("CBS", "M*", "Voronoi")
MARL_ALGORITHMS = ("VDN", "QMIX", "QTRAN", "QPLEX")
ACTION_NAMES: tuple[Action, ...] = ("stay", "up", "down", "left", "right")

# How long (in milliseconds) between simulation steps.
SIMULATION_INTERVAL_MS = 600

# Used to create a random seed when the player does not type one in.
SEED_GENERATOR = SystemRandom()

# Text shown on each difficulty button.
DIFFICULTIES = {
    1: {
        "title": "      Level 1",
        "description": "5 x 5 grid | 2 pursuers | no objects",
    },
    2: {
        "title": "      Level 2",
        "description": "10 x 10 grid | 3 pursuers | walls and barrels",
    },
    3: {
        "title": "      Level 3",
        "description": "20 x 20 grid | 4 pursuers | many walls and barrels",
    },
}

# Colours used by the interface, gathered in one place so they are easy to edit.
COLOR_BACKGROUND = "#111827"
COLOR_TEXT_MAIN = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_TEXT_LIGHT = "#cbd5e1"
COLOR_ERROR = "#fca5a5"
COLOR_SUCCESS = "#22c55e"
COLOR_FAILURE = "#f87171"
COLOR_PRIMARY = "#1c0056"
COLOR_PRIMARY_HOVER = "#2d0080"
COLOR_PANEL = "#ffffff"
COLOR_PANEL_TEXT = "#111827"
COLOR_MUTED = "#6b7280"


# PART 1A - BOARD HELPERS
def render_ascii(environment: PursuitEvasionEnv) -> str:
    """
    Return the board as plain text, which is handy for debugging.

    Symbols:  . empty   B barrel   G goal   E evader   P pursuer

    Layers are drawn from bottom to top, so agents are always visible
    on top of barrels and goals.
    """
    state = environment.observe()
    scenario = environment.scenario

    # Start with an empty grid of dots.
    grid = []
    for _ in range(scenario.rows):
        grid.append(["."] * scenario.cols)

    # Layer 1: barrels.
    for row, col in scenario.barrels:
        grid[row][col] = "B"

    # Layer 2: goals.
    for row, col in scenario.goals:
        grid[row][col] = "G"

    # Layer 3: evaders.
    for row, col in state["evaders"]:
        grid[row][col] = "E"

    # Layer 4: pursuers (drawn last so they show on top).
    for row, col in state["pursuers"]:
        grid[row][col] = "P"

    # Join cells with spaces, and rows with newlines.
    lines = []
    for row in grid:
        lines.append(" ".join(row))
    return "\n".join(lines)


def _distance(first: Position, second: Position) -> int:
    """Manhattan distance: how many up/down/left/right steps apart two cells are."""
    row_difference = abs(first[0] - second[0])
    col_difference = abs(first[1] - second[1])
    return row_difference + col_difference


def _nearest_target(position: Position, targets: tuple[Position, ...]) -> Position:
    """Return whichever target is closest to the given position."""
    return min(targets, key=lambda target: _distance(position, target))


def _position_after_action(
    environment: PursuitEvasionEnv,
    position: Position,
    action: Action,
    speed: int = 1,
) -> Position:
    """
    Ask the environment where an agent would end up after taking an action.

    The environment handles walls, barrels and board edges. If the move is
    blocked, the same position is returned.
    """
    return environment._move(position, action, speed)


def _legal_actions(
    environment: PursuitEvasionEnv,
    position: Position,
) -> list[Action]:
    """Return every movement action that actually changes the agent's position."""
    legal = []
    for action in MOVEMENT_ACTIONS:
        new_position = _position_after_action(environment, position, action)
        if new_position != position:
            legal.append(action)
    return legal


# PART 1B - PATH FINDING

def _first_step_to_target(
    environment: PursuitEvasionEnv,
    start: Position,
    target: Position,
    blocked_positions: set[Position] | None = None,
) -> Action | None:
    """
    Find the shortest legal route from start to target using breadth-first
    search (BFS), and return ONLY the first action of that route.

    We only need the first step because the agents re-plan on every timestep.
    Returns None if no route exists.

    blocked_positions are cells the route must avoid (for example, cells the
    pursuers are expected to occupy).
    """
    blocked = blocked_positions or set()

    # Each queue entry is (cell, the first action taken to reach that cell).
    queue: deque[tuple[Position, Action | None]] = deque()
    queue.append((start, None))

    visited = {start}

    while queue:
        position, first_action = queue.popleft()

        # Reached the target: report the first move of the route.
        if position == target:
            return first_action

        # Otherwise explore each neighbouring cell.
        for action in MOVEMENT_ACTIONS:
            candidate = _position_after_action(environment, position, action, 1)

            # Skip cells we cannot move into, already saw, or must avoid.
            if candidate == position:
                continue
            if candidate in visited:
                continue
            if candidate in blocked:
                continue

            visited.add(candidate)

            # Remember the very first action of this route. For the start
            # cell it is None, so the first move taken becomes the label.
            if first_action is None:
                route_first_action = action
            else:
                route_first_action = first_action

            queue.append((candidate, route_first_action))

    # The queue emptied without reaching the target.
    return None


def _fallback_movement_action(
    environment: PursuitEvasionEnv,
    position: Position,
    targets: tuple[Position, ...] = (),
) -> Action:
    """
    Pick a sensible move when path search found no route.

    - No legal moves at all: return "up" (the move will simply be blocked).
    - No targets given: return the first legal move.
    - Otherwise: return the legal move that ends closest to any target.
    """
    legal = _legal_actions(environment, position)

    if not legal:
        return MOVEMENT_ACTIONS[0]

    if not targets:
        return legal[0]

    def distance_to_closest_target(action: Action) -> int:
        new_position = _position_after_action(environment, position, action)
        return min(_distance(new_position, target) for target in targets)

    return min(legal, key=distance_to_closest_target)


# PART 1C - AGENT DECISIONS

def _pursuer_action(environment: PursuitEvasionEnv, pursuer: Position) -> Action:
    """
    Decide a pursuer's move: walk the shortest path toward the nearest evader.
    If no path exists, fall back to a simple greedy move.
    """
    evaders = tuple(environment.state.evaders)
    target = _nearest_target(pursuer, evaders)

    path_action = _first_step_to_target(environment, pursuer, target)
    if path_action is not None:
        return path_action

    return _fallback_movement_action(environment, pursuer, (target,))


def _evader_action(
    environment: PursuitEvasionEnv,
    evader: Position,
    pursuer_positions: set[Position],
) -> Action:
    """
    Decide an evader's move.

    Step 1 (goal modes only): try to find a path to the nearest goal that
            avoids every cell in pursuer_positions. If one exists, take it.

    Step 2 (fallback, and the only step in Pure Pursuit): choose the legal
            move that keeps the evader as far from the pursuers as possible.
            In goal modes, ties are broken by preferring moves that get
            closer to the nearest goal.
    """
    goal_mode = environment.mode in (GameMode.REACH_AVOID, GameMode.MAZE)

    # Step 1: safe route to a goal
    if goal_mode:
        nearest_goal = _nearest_target(evader, environment.scenario.goals)
        safe_action = _first_step_to_target(
            environment, evader, nearest_goal, pursuer_positions
        )
        if safe_action is not None:
            return safe_action

    # Step 2: run away from the pursuers
    legal = _legal_actions(environment, evader)
    if not legal:
        return MOVEMENT_ACTIONS[0]

    def score_action(action: Action) -> tuple[int, int]:
        """
        Higher score is better. Python compares the two numbers in order:
        the first number is compared first, and the second only breaks ties.
        """
        new_position = _position_after_action(environment, evader, action)

        # First priority: distance from the closest pursuer (bigger = safer).
        safety = min(_distance(new_position, pursuer) for pursuer in pursuer_positions)

        # Second priority (goal modes only): be closer to the nearest goal.
        # It is negative because a smaller distance should give a higher score.
        if goal_mode:
            nearest_goal = _nearest_target(evader, environment.scenario.goals)
            goal_bonus = -_distance(new_position, nearest_goal)
        else:
            goal_bonus = 0

        return (safety, goal_bonus)

    return max(legal, key=score_action)


def _planner_scenario(environment: PursuitEvasionEnv):
    return replace(
        environment.scenario,
        pursuers=tuple(environment.state.pursuers),
        evaders=tuple(environment.state.evaders),
    )


def _marl_pursuer_actions(
    environment: PursuitEvasionEnv,
    algorithm: str,
) -> list[Action]:
    """Run a saved MARL policy, falling back clearly when no checkpoint exists."""
    checkpoint = Path("checkpoints") / f"{algorithm.lower()}_{environment.scenario.seed}.pt"
    if not checkpoint.exists():
        return [
            _pursuer_action(environment, pursuer)
            for pursuer in environment.state.pursuers
        ]
    import torch
    from marl.base import AgentNetwork, state_tensor

    observation = environment.observe()
    policy = AgentNetwork(len(observation["global_state"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["agents"])
    policy.eval()
    with torch.no_grad():
        values = policy(state_tensor(observation)).repeat(len(environment.state.pursuers), 1)
    actions: list[Action] = []
    for index, mask in enumerate(observation["action_masks"]["pursuers"]):
        masked = values[index].clone()
        masked[~torch.tensor(mask, dtype=torch.bool)] = -torch.inf
        actions.append(ACTION_NAMES[int(masked.argmax())])
    return actions


def _planner_action(value: object, fallback: Action) -> Action:
    """Validate an untyped planner action at the UI/environment boundary."""
    if isinstance(value, str) and value in ACTION_NAMES:
        return cast(Action, value)
    return fallback


def _autonomous_actions(
    environment: PursuitEvasionEnv,
    algorithm: str = "CBS",
) -> tuple[list[Action], list[Action]]:
    """
    Compute one move for every pursuer and every evader this timestep.

    Order matters here:
      1. Pursuers choose first.
      2. We predict where those pursuers will be (current AND next cell).
      3. Evaders choose while avoiding all of those predicted cells.
    """
    # 1. Pursuer moves. Classical planners are receding-horizon controllers:
    # plan from the current state, then execute only each route's first action.
    pursuer_actions: list[Action]
    if algorithm in {"CBS", "M*", "Voronoi"}:
        planner = {"CBS": cbs_plan, "M*": mstar_plan, "Voronoi": voronoi_plan}[algorithm]
        result = planner(_planner_scenario(environment))
        pursuer_actions = [
            _planner_action(
                path[0] if path else None,
                _pursuer_action(environment, pursuer),
            )
            for pursuer, path in zip(environment.state.pursuers, result["paths"])
        ]
    else:
        pursuer_actions = _marl_pursuer_actions(environment, algorithm)

    # 2. Predict danger cells: where each pursuer is now and where it will land.
    predicted_pursuer_positions: set[Position] = set()
    for pursuer, action in zip(environment.state.pursuers, pursuer_actions):
        next_position = _position_after_action(
            environment, pursuer, action, environment.scenario.pursuer_speed
        )
        predicted_pursuer_positions.add(pursuer)
        predicted_pursuer_positions.add(next_position)

    # 3. Evader moves.
    evader_actions: list[Action] = []
    for evader in environment.state.evaders:
        evader_actions.append(
            _evader_action(environment, evader, predicted_pursuer_positions)
        )

    return pursuer_actions, evader_actions


# PART 2 - THE WINDOW (Tkinter app)

class PursuitEvasionApp:
    """
    The whole application: menus, drawing, and the simulation loop.

    Screens (in order):
        Main menu -> Scenario Configuration -> Scenario
        Main menu -> Simulation Configuration -> Simulation Results
    """

    # Setup

    def __init__(self, initial_level: int = 1, seed: int = 7):
        """Store default settings, create the window, and show the main menu."""

        # Menu selections (updated as the player clicks through the menus).
        self.seed = seed
        self.level = initial_level
        self.mode = GameMode.PURE_PURSUIT
        self.algorithm = "CBS"
        self.selected_seed: int | None = None  # None means "pick a random seed"

        # Simulation state.
        self.environment: PursuitEvasionEnv | None = None
        self.running = False
        self.paused = False

        # Timer values for the on-screen clock.
        self.started_at = 0.0  # when the clock last started counting
        self.elapsed = 0.0     # seconds accumulated before started_at

        # Create the window.
        self.root = ctk.CTk()
        self._configure_styles()
        self.root.title("Multi-Agent Strategies for \nPursuit-Evasion Games")
        self.root.configure(fg_color="#f3f4f6")
        self.root.minsize(900, 650)

        self.show_main_menu()

    def _configure_styles(self) -> None:
        """Define the look of the blue "Action" buttons used everywhere."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "Action.TButton",
            anchor="center",
            justify="center",
            font=("Segoe UI", 11, "bold"),
            padding=(18, 11),
            foreground=COLOR_TEXT_MAIN,
            background="#2563eb",
            borderwidth=0,
        )

        # Colours for hover, click, and disabled states.
        style.map(
            "Action.TButton",
            background=[("active", "#3b82f6"), ("pressed", "#1d4ed8")],
            foreground=[("disabled", COLOR_TEXT_MUTED)],
        )

    # Small UI building blocks (used by every menu)

    def _clear(self) -> None:
        """Remove every widget from the window so a new screen can be built."""
        for child in self.root.winfo_children():
            child.destroy()

    def _frame(self) -> ctk.CTkFrame:
        """Create the padded container that holds a menu screen."""
        frame = ctk.CTkFrame(self.root, fg_color="#f3f4f6")
        frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        return frame

    def _title(self, parent: tk.Misc, text: str, subtitle: str) -> None:
        """Add a large heading with a smaller grey subtitle underneath."""
        heading = ctk.CTkLabel(
            parent,
            text=text,
            text_color="#1e1e2f",
            font=ctk.CTkFont(family="Segoe UI", size=25, weight="bold"),
        )
        heading.pack(pady=(25, 8))

        sub_heading = ctk.CTkLabel(
            parent,
            text=subtitle,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        sub_heading.pack(pady=(0, 30))

    def _button(self, parent: tk.Misc, text: str, command, width: int = 28) -> None:
        """Add a blue button that runs `command` when clicked."""
        button = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=42,
            corner_radius=10,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        button.pack(pady=7)

    # Menu screens

    def show_main_menu(self) -> None:
        """Show the single entry point for game and batch simulation setup."""
        self.running = False
        self._clear()
        frame = self._frame()

        self._title(
            frame,
            "Multi-Agent Strategies for \nPursuit-Evasion Games",
            "By Zack Haber, 2027",
        )
        self._button(frame, "Play Game", self.show_scenario_config)
        self._button(frame, "Simulation", self.show_simulation_config)
        self._button(frame, "Quit", self.root.destroy)

    def _configuration_panel(self, title: str, subtitle: str) -> tuple[ctk.CTkFrame, ctk.CTkFrame]:
        """Create the testUI-style white configuration card and its body."""
        self._clear()
        outer = self._frame()
        card = ctk.CTkFrame(
            outer,
            corner_radius=20,
            fg_color=COLOR_PANEL,
            border_color="#e5e7eb",
            border_width=1,
        )
        card.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        ctk.CTkLabel(
            card,
            text=title,
            text_color="#1e1e2f",
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        ).pack(fill=tk.X, padx=30, pady=(25, 2))
        ctk.CTkLabel(
            card,
            text=subtitle,
            text_color=COLOR_MUTED,
            anchor="w",
            font=ctk.CTkFont(size=13),
        ).pack(fill=tk.X, padx=30, pady=(0, 20))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill=tk.BOTH, expand=True, padx=30)
        return outer, body

    def _config_label(self, parent: ctk.CTkFrame, text: str) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            text_color=COLOR_MUTED,
            anchor="w",
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(fill=tk.X, pady=(0, 6))

    def _config_dropdown(self, parent: ctk.CTkFrame, values: tuple[str, ...]) -> ctk.CTkOptionMenu:
        dropdown = ctk.CTkOptionMenu(
            parent,
            values=list(values),
            fg_color=COLOR_PANEL,
            text_color=COLOR_PANEL_TEXT,
            button_color=COLOR_PANEL,
            button_hover_color="#f3f4f6",
            dropdown_fg_color=COLOR_PANEL,
            dropdown_text_color=COLOR_PANEL_TEXT,
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        dropdown.pack(fill=tk.X, pady=(0, 14))
        return dropdown

    def _config_footer(
        self, parent: ctk.CTkFrame, run_text: str, command, cancel_text: str = "Cancel"
    ) -> None:
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(12, 25))
        run_button = ctk.CTkButton(
            footer,
            text=run_text,
            command=command,
            width=150,
            height=42,
            corner_radius=10,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        run_button.pack(side=tk.RIGHT, padx=(8, 0))
        ctk.CTkButton(
            footer,
            text=cancel_text,
            command=self.show_main_menu,
            width=90,
            height=42,
            corner_radius=10,
            fg_color=COLOR_PANEL,
            hover_color="#f3f4f6",
            text_color=COLOR_PANEL_TEXT,
            border_color="#d1d5db",
            border_width=1,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side=tk.RIGHT)

    def _configuration_fields(self, parent: ctk.CTkFrame, simulation: bool = False) -> None:
        """Build the complete testUI scenario form inside the current card."""
        parent.grid_columnconfigure(0, weight=1)
        self._config_label(parent, "GAME MODE")
        self.mode_dropdown = self._config_dropdown(
            parent, ("Maze", "Reach and Avoid", "Pure Pursuit")
        )
        self.mode_dropdown.configure(command=self._on_mode_change)
        ctk.CTkLabel(
            parent,
            text="Maze · Reach and Avoid · Pure Pursuit",
            text_color="#9ca3af",
            anchor="w",
            font=ctk.CTkFont(size=11),
        ).pack(fill=tk.X, pady=(0, 16))

        self._config_label(parent, "ALGORITHM")
        self.family_toggle = ctk.CTkSegmentedButton(
            parent,
            values=["Planner", "MARL"],
            command=self._on_family_change,
            fg_color="#f3f4f6",
            selected_color=COLOR_PRIMARY,
            selected_hover_color=COLOR_PRIMARY_HOVER,
            unselected_color="#4b5563",
            unselected_hover_color="#6b7280",
            text_color="#ffffff",
            corner_radius=10,
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.family_toggle.set("Planner")
        self.family_toggle.pack(fill=tk.X, pady=(0, 16))

        self.algo_section_label = ctk.CTkLabel(
            parent,
            text="PLANNER ALGORITHM",
            text_color=COLOR_MUTED,
            anchor="w",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.algo_section_label.pack(fill=tk.X, pady=(0, 6))
        self.algo_dropdown = self._config_dropdown(parent, CLASSICAL_ALGORITHMS)
        ctk.CTkLabel(
            parent,
            text="Planner: CBS, Voronoi, M*\nMARL: QTRAN, QPLEX, QMIX, VDN\nOnly the selected family's algorithms are shown above.",
            text_color="#9ca3af",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11),
        ).pack(fill=tk.X, pady=(0, 18))

        self._config_label(parent, "SCENARIO SIZE (X = Y)")
        size_frame = ctk.CTkFrame(parent, fg_color="transparent")
        size_frame.pack(fill=tk.X, pady=(0, 18))
        self.size_entry = ctk.CTkEntry(
            size_frame,
            placeholder_text="20",
            fg_color=COLOR_PANEL,
            text_color=COLOR_PANEL_TEXT,
            border_color="#d1d5db",
            border_width=1,
            corner_radius=10,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.size_entry.insert(0, "20")
        self.size_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 12))
        self.size_entry.bind("<KeyRelease>", self._update_size_badge)
        ctk.CTkLabel(
            self.size_entry,
            text="units",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=12),
        ).place(relx=0.92, rely=0.5, anchor="e")
        self.badge_card = ctk.CTkFrame(
            size_frame,
            fg_color=COLOR_PANEL,
            border_color="#e5e7eb",
            border_width=1,
            corner_radius=10,
            width=130,
            height=45,
        )
        self.badge_card.pack(side=tk.RIGHT)
        self.badge_card.pack_propagate(False)
        ctk.CTkLabel(
            self.badge_card,
            text="Resulting size",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", padx=10, pady=(4, 0))
        self.badge_val = ctk.CTkLabel(
            self.badge_card,
            text="20×20",
            text_color=COLOR_PANEL_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.badge_val.pack(anchor="w", padx=10, pady=(0, 4))

        self._config_label(parent, "OBJECT DENSITY")
        self.walls_density_val = 0.5
        self.barrels_density_val = 0.0
        self.walls_row, self.walls_val_label, _ = self._create_stepper_row_control(
            parent, "Walls Density", self.walls_density_val, self._on_walls_change
        )
        self.walls_row.pack(fill=tk.X, pady=(0, 8))
        self.barrels_row, self.barrels_val_label, self.barrel_btns = (
            self._create_stepper_row_control(
                parent, "Barrels Density", self.barrels_density_val,
                self._on_barrels_change,
            )
        )
        self.barrels_row.pack(fill=tk.X, pady=(0, 8))
        self.info_banner = ctk.CTkLabel(
            parent,
            text="ⓘ  Barrels are unavailable in Maze mode, so density is fixed at 0.",
            text_color="#9ca3af",
            anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self.info_banner.pack(fill=tk.X, pady=(0, 25))
        self._on_mode_change("Maze")

        if simulation:
            self._config_label(parent, "NUMBER OF SIMULATIONS")
            self.config_runs = ctk.CTkEntry(
                parent,
                placeholder_text="10",
                height=42,
                corner_radius=10,
                border_width=1,
                border_color="#d1d5db",
                fg_color=COLOR_PANEL,
                text_color=COLOR_PANEL_TEXT,
            )
            self.config_runs.insert(0, "10")
            self.config_runs.pack(fill=tk.X, pady=(0, 14))

    def _on_family_change(self, family: str) -> None:
        algorithms = CLASSICAL_ALGORITHMS if family == "Planner" else MARL_ALGORITHMS
        self.algo_section_label.configure(
            text="PLANNER ALGORITHM" if family == "Planner" else "MARL ALGORITHM"
        )
        self.algo_dropdown.configure(values=list(algorithms))
        self.algo_dropdown.set(algorithms[0])

    def _on_config_family_change(self, family: str) -> None:
        self._on_family_change(family)

    def _on_mode_change(self, choice: str) -> None:
        if choice == "Maze":
            self.barrels_val_label.configure(text="0.0", text_color="#d1d5db")
            for button in self.barrel_btns:
                button.configure(
                    state="disabled", fg_color="#f9fafb", text_color="#d1d5db"
                )
            self.info_banner.pack(fill=tk.X, pady=(0, 25))
        else:
            self.barrels_val_label.configure(
                text=f"{self.barrels_density_val:.1f}", text_color=COLOR_PANEL_TEXT
            )
            for button in self.barrel_btns:
                button.configure(
                    state="normal", fg_color=COLOR_PANEL, text_color=COLOR_PANEL_TEXT
                )
            self.info_banner.pack_forget()

    def _update_size_badge(self, _event=None) -> None:
        value = self.size_entry.get().strip()
        if value.isdigit():
            self.badge_val.configure(text=f"{value}×{value}")

    def _create_stepper_row(self, parent, title, initial_value, callback):
        row, _, _ = self._create_stepper_row_control(
            parent, title, initial_value, callback
        )
        return row

    def _create_stepper_row_control(self, parent, title, initial_value, callback):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(
            row,
            text=title,
            text_color=COLOR_PANEL_TEXT,
            anchor="w",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        stepper = ctk.CTkFrame(row, fg_color="transparent")
        stepper.pack(side=tk.RIGHT)
        buttons = []
        for text, delta in (("-", -0.1), ("+", 0.1)):
            button = ctk.CTkButton(
                stepper,
                text=text,
                width=32,
                height=32,
                fg_color=COLOR_PANEL,
                hover_color="#f3f4f6",
                text_color=COLOR_PANEL_TEXT,
                border_color="#e5e7eb",
                border_width=1,
                corner_radius=6,
                command=lambda change=delta: callback(change),
                font=ctk.CTkFont(size=14, weight="bold"),
            )
            buttons.append(button)
        buttons[0].pack(side=tk.LEFT, padx=2)
        value_box = ctk.CTkFrame(
            stepper,
            fg_color=COLOR_PANEL,
            border_color="#e5e7eb",
            border_width=1,
            corner_radius=6,
            width=50,
            height=32,
        )
        value_box.pack(side=tk.LEFT, padx=2)
        value_box.pack_propagate(False)
        value_label = ctk.CTkLabel(
            value_box,
            text=f"{initial_value:.1f}",
            text_color=COLOR_PANEL_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        value_label.place(relx=0.5, rely=0.5, anchor="center")
        buttons[1].pack(side=tk.LEFT, padx=2)
        return row, value_label, buttons

    def _on_walls_change(self, delta: float) -> None:
        self.walls_density_val = round(
            max(0.0, min(1.0, self.walls_density_val + delta)), 1
        )
        self.walls_val_label.configure(text=f"{self.walls_density_val:.1f}")

    def _on_barrels_change(self, delta: float) -> None:
        if self.mode_dropdown.get() != "Maze":
            self.barrels_density_val = round(
                max(0.0, min(1.0, self.barrels_density_val + delta)), 1
            )
            self.barrels_val_label.configure(text=f"{self.barrels_density_val:.1f}")

    def _read_configuration(self, simulation: bool = False) -> tuple[int, int | None] | None:
        """Validate the shared form and return (level, seed)."""
        size_text = self.size_entry.get().strip()
        if not size_text.isdigit() or int(size_text) < 5:
            self.size_entry.configure(border_color=COLOR_FAILURE)
            return None
        size = int(size_text)
        seed_text = ""
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            return None
        if simulation:
            try:
                runs = int(self.config_runs.get().strip())
            except ValueError:
                self.config_runs.configure(border_color=COLOR_FAILURE)
                return None
            if runs < 1 or runs > 10000:
                self.config_runs.configure(border_color=COLOR_FAILURE)
                return None
            self.simulation_runs = runs
        self.mode = {
            "Pure Pursuit": GameMode.PURE_PURSUIT,
            "Reach and Avoid": GameMode.REACH_AVOID,
            "Maze": GameMode.MAZE,
        }[self.mode_dropdown.get()]
        self.algorithm = self.algo_dropdown.get()
        self.custom_grid_size = size
        self.custom_wall_density = self.walls_density_val
        self.custom_barrel_density = self.barrels_density_val
        level = 1 if size <= 7 else 2 if size <= 14 else 3
        return level, seed

    def show_scenario_config(self) -> None:
        """Show the consolidated Play Game configuration panel."""
        _, body = self._configuration_panel(
            "Scenario Configuration",
            "Set up your environment and algorithm parameters.",
        )
        self._configuration_fields(body)
        self._config_footer(body, "Run Scenario", self._start_configured_scenario)

    def _start_configured_scenario(self) -> None:
        configuration = self._read_configuration()
        if configuration is not None:
            level, seed = configuration
            self.start_simulation(level, seed)

    def show_simulation_config(self) -> None:
        """Show batch simulation configuration and its results screen."""
        _, body = self._configuration_panel(
            "Simulation Configuration",
            "Run the selected scenario repeatedly and measure its performance.",
        )
        self._configuration_fields(body, simulation=True)
        self._config_footer(
            body, "Run Simulations", self._start_batch_simulation, "Quit"
        )

    def _start_batch_simulation(self) -> None:
        configuration = self._read_configuration(simulation=True)
        if configuration is None:
            return
        level, seed = configuration
        self._show_batch_progress()
        Thread(
            target=self._run_batch_worker,
            args=(level, seed),
            daemon=True,
        ).start()

    def _show_batch_progress(self) -> None:
        """Keep Tk responsive while the batch worker evaluates episodes."""
        _, body = self._configuration_panel(
            "Simulation Running",
            "Please wait while the configured simulations are evaluated.",
        )
        ctk.CTkLabel(
            body,
            text=f"Running {self.simulation_runs} simulation(s)...",
            text_color=COLOR_PANEL_TEXT,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=40)
        ctk.CTkLabel(
            body,
            text="The window remains responsive while results are calculated.",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(size=12),
        ).pack()

    def _run_batch_worker(self, level: int, seed: int | None) -> None:
        try:
            self.run_batch_simulations(level, seed)
        except Exception as error:
            self.root.after(0, lambda: self._show_batch_error(error))

    def run_batch_simulations(self, level: int, seed: int | None) -> None:
        """Run the configured policy repeatedly and show measurable results."""
        base_seed = (
            seed if seed is not None else SEED_GENERATOR.randrange(0, 2**32)
        )
        capture_count = 0
        capture_steps: list[int] = []
        planning_times: list[float] = []
        episode_times: list[float] = []

        for index in range(self.simulation_runs):
            episode_seed = (base_seed + index) % (2**32)
            scenario = generate_level(
                DifficultySpec(
                    level=level,
                    grid_size=getattr(self, "custom_grid_size", None),
                    wall_density=getattr(self, "custom_wall_density", None),
                    barrel_density=getattr(self, "custom_barrel_density", None),
                    seed=episode_seed,
                ),
                self.mode,
            )
            environment = PursuitEvasionEnv(
                scenario, self.mode, seed=episode_seed
            )
            environment.reset()
            episode_started = perf_counter()
            last_info: dict = {"captured": False, "time": 0}
            while not environment.done:
                planning_started = perf_counter()
                pursuer_actions, evader_actions = _autonomous_actions(
                    environment, self.algorithm
                )
                planning_times.append(perf_counter() - planning_started)
                _, _, _, _, last_info = environment.step(
                    pursuer_actions, evader_actions
                )
            episode_times.append(perf_counter() - episode_started)
            if bool(last_info.get("captured")):
                capture_count += 1
                capture_steps.append(int(last_info.get("time", 0)))

        success_rate = capture_count / self.simulation_runs * 100
        average_steps = (
            sum(capture_steps) / len(capture_steps) if capture_steps else 0.0
        )
        average_planning_ms = (
            sum(planning_times) / len(planning_times) * 1000
            if planning_times
            else 0.0
        )
        average_time_ms = (
            sum(episode_times) / len(episode_times) * 1000
            if episode_times
            else 0.0
        )
        metric_label = (
            "Planning time per decision"
            if self.algorithm in CLASSICAL_ALGORITHMS
            else "Inference time per decision"
        )
        self.root.after(
            0,
            lambda: self._show_batch_results(
                level,
                base_seed,
                success_rate,
                average_steps,
                average_planning_ms,
                average_time_ms,
                metric_label,
            ),
        )

    def _show_batch_error(self, error: Exception) -> None:
        """Show a recoverable batch error instead of closing the application."""
        _, body = self._configuration_panel(
            "Simulation Error",
            "The simulation could not be completed.",
        )
        ctk.CTkLabel(
            body,
            text=str(error),
            text_color=COLOR_FAILURE,
            wraplength=650,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=13),
        ).pack(fill=tk.X, pady=30)
        self._config_footer(body, "Try Again", self.show_simulation_config, "Quit")

    def _show_batch_results(
        self,
        level: int,
        seed: int,
        success_rate: float,
        average_steps: float,
        average_planning_ms: float,
        average_time_ms: float,
        metric_label: str,
    ) -> None:
        """Render batch statistics in the same card style as configuration."""
        _, body = self._configuration_panel(
            "Simulation Results",
            f"{self.algorithm} · {MODE_LABELS[self.mode]} · Level {level} · Seed {seed}",
        )
        metrics = (
            ("Success Rate (%)", f"{success_rate:.1f}%"),
            ("Path Efficiency (Steps to Capture)", f"{average_steps:.1f}"),
            ("Computation / Latency Cost", f"{average_planning_ms:.3f} ms"),
            ("Average Time", f"{average_time_ms:.3f} ms"),
        )
        for label, value in metrics:
            row = ctk.CTkFrame(
                body,
                fg_color="#f9fafb",
                corner_radius=10,
                height=52,
            )
            row.pack(fill=tk.X, pady=5)
            row.pack_propagate(False)
            ctk.CTkLabel(
                row,
                text=label,
                text_color=COLOR_MUTED,
                anchor="w",
                font=ctk.CTkFont(size=12),
            ).pack(side=tk.LEFT, padx=14)
            ctk.CTkLabel(
                row,
                text=value,
                text_color=COLOR_PANEL_TEXT,
                anchor="e",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(side=tk.RIGHT, padx=14)
        ctk.CTkLabel(
            body,
            text=f"{metric_label}: measured across every decision in all simulations.",
            text_color=COLOR_MUTED,
            anchor="w",
            font=ctk.CTkFont(size=11),
        ).pack(fill=tk.X, pady=(12, 0))
        self._config_footer(body, "Run Again", self.show_simulation_config)

    def show_mode_menu(self) -> None:
        """Screen 2: choose the game mode."""
        self._clear()
        frame = self._frame()

        self._title(frame, "Choose Game Mode", "Select the objective for the evader")

        self._button(
            frame,
            "Pure Pursuit\n(Survive until captured)",
            lambda: self.choose_mode(GameMode.PURE_PURSUIT),
        )
        self._button(
            frame,
            "Reach and Avoid\n(Reach a goal before capture)",
            lambda: self.choose_mode(GameMode.REACH_AVOID),
        )
        self._button(
            frame,
            "Maze\n(Navigate walls to goal or capture)",
            lambda: self.choose_mode(GameMode.MAZE),
        )
        self._button(frame, "Back", self.show_main_menu)

    def choose_mode(self, mode: GameMode) -> None:
        """Remember the chosen mode and move on to the algorithm menu."""
        self.mode = mode
        self.show_algorithm_menu()

    def show_algorithm_menu(self) -> None:
        """Screen 3: choose an algorithm, in two columns (classical vs MARL)."""
        self._clear()
        frame = self._frame()

        self._title(frame, "Choose Algorithm", f"Mode: {MODE_LABELS[self.mode]}")

        # A row that will hold the two columns side by side.
        columns = tk.Frame(frame, bg=COLOR_BACKGROUND)
        columns.pack(fill=tk.X, expand=True, pady=(0, 18))

        # Left column.
        classical_column = tk.Frame(columns, bg=COLOR_BACKGROUND)
        classical_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        # Right column.
        marl_column = tk.Frame(columns, bg=COLOR_BACKGROUND)
        marl_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))

        # Left column contents.
        classical_heading = tk.Label(
            classical_column,
            text="Classical Planners",
            fg="#93c5fd",
            bg=COLOR_BACKGROUND,
            font=("Segoe UI", 11, "bold"),
        )
        classical_heading.pack(pady=(0, 4))

        for algorithm in CLASSICAL_ALGORITHMS:
            # "selected=algorithm" captures the current value for this button.
            self._button(
                classical_column,
                algorithm,
                lambda selected=algorithm: self.choose_algorithm(selected),
                18,
            )

        # Right column contents.
        marl_heading = tk.Label(
            marl_column,
            text="MARL Algorithms",
            fg="#86efac",
            bg=COLOR_BACKGROUND,
            font=("Segoe UI", 11, "bold"),
        )
        marl_heading.pack(pady=(0, 4))

        for algorithm in MARL_ALGORITHMS:
            self._button(
                marl_column,
                algorithm,
                lambda selected=algorithm: self.choose_algorithm(selected),
                18,
            )

        self._button(frame, "Back", self.show_mode_menu)

    def choose_algorithm(self, algorithm: str) -> None:
        """Remember the chosen algorithm and move on to the seed menu."""
        self.algorithm = algorithm
        self.show_seed_menu()

    def show_seed_menu(self) -> None:
        """Screen 4: use a random seed, or type a specific one."""
        self._clear()
        frame = self._frame()

        self._title(
            frame,
            "Choose Seed",
            f"Mode: {MODE_LABELS[self.mode]} | Algorithm: {self.algorithm}",
        )

        self._button(frame, "Random Seed", self.use_random_seed)

        # Container for the "type your own seed" section.
        seed_entry = tk.Frame(frame, bg=COLOR_BACKGROUND)
        seed_entry.pack(pady=(18, 4))

        seed_label = tk.Label(
            seed_entry,
            text="Specific Seed",
            fg=COLOR_TEXT_LIGHT,
            bg=COLOR_BACKGROUND,
            font=("Segoe UI", 11),
        )
        seed_label.pack(pady=(0, 6))

        # Text box where the player types a seed.
        self.seed_input = ttk.Entry(seed_entry, width=28, justify="center")
        self.seed_input.pack(pady=(0, 6))
        self.seed_input.bind("<Return>", lambda _event: self.use_specific_seed())

        # Red error message (empty until the player types something invalid).
        self.seed_error = tk.Label(
            seed_entry,
            text="",
            fg=COLOR_ERROR,
            bg=COLOR_BACKGROUND,
            font=("Segoe UI", 10),
        )
        self.seed_error.pack()

        self._button(seed_entry, "Search", self.use_specific_seed, 18)
        self._button(frame, "Back", self.show_algorithm_menu)

        # Put the cursor in the text box so the player can type immediately.
        self.seed_input.focus_set()

    def use_random_seed(self) -> None:
        """Choose "random" (stored as None) and move on to difficulty."""
        self.selected_seed = None
        self.show_difficulty_menu()

    def use_specific_seed(self) -> None:
        """Read the typed seed. If it is not a whole number, show an error."""
        typed_text = self.seed_input.get().strip()

        try:
            self.selected_seed = int(typed_text)
        except ValueError:
            self.seed_error.config(text="Enter a valid integer seed")
            self.seed_input.focus_set()
            return

        self.show_difficulty_menu()

    def show_difficulty_menu(self) -> None:
        """Screen 5: choose a level. Clicking a level starts the simulation."""
        self._clear()
        frame = self._frame()

        self._title(
            frame,
            "Choose Difficulty",
            f"Mode: {MODE_LABELS[self.mode]} | Algorithm: {self.algorithm}",
        )

        for level, details in DIFFICULTIES.items():
            button_text = f"{details['title']}\n{details['description']}"
            self._button(
                frame,
                button_text,
                lambda selected=level: self.start_simulation(
                    selected, self.selected_seed
                ),
                45,
            )

        self._button(frame, "Back", self.show_seed_menu)

    # Starting a simulation

    def _new_environment(self) -> PursuitEvasionEnv:
        """Build a fresh environment from the current level, seed and mode."""
        difficulty = DifficultySpec(
            level=self.level,
            grid_size=getattr(self, "custom_grid_size", None),
            wall_density=getattr(self, "custom_wall_density", None),
            barrel_density=getattr(self, "custom_barrel_density", None),
            seed=self.seed,
        )
        scenario = generate_level(difficulty, self.mode)

        environment = PursuitEvasionEnv(scenario, self.mode)
        environment.reset()
        return environment

    def start_simulation(self, level: int, seed: int | None = None) -> None:
        """
        Begin a run.

        If seed is None, a random 32-bit seed is generated. The seed is shown
        in the side panel so a run can be repeated later with "Restart".
        """
        self.level = level

        if seed is not None:
            self.seed = seed
        else:
            self.seed = SEED_GENERATOR.randrange(0, 2**32)

        self.environment = self._new_environment()

        # Reset the clock and state flags.
        self.elapsed = 0.0
        self.started_at = perf_counter()
        self.paused = False
        self.running = True

        # Build the screen, then schedule the first timer tick.
        self._build_simulation_screen()
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _build_simulation_screen(self) -> None:
        """Create the board canvas (left) and the status/control panel (right)."""
        self._clear()

        # Left side: the board
        self.canvas = tk.Canvas(self.root, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(18, 8), pady=18)

        # Right side: the panel
        panel = tk.Frame(self.root, bg=COLOR_BACKGROUND, width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 18), pady=18)
        panel.pack_propagate(False)  # keep the panel at its fixed width

        panel_title = tk.Label(
            panel,
            text="SIMULATION",
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BACKGROUND,
            font=("Segoe UI", 16, "bold"),
        )
        panel_title.pack(anchor="w", pady=(8, 15))

        # Label that shows mode, algorithm, seed, time, step, etc.
        self.run_status = tk.Label(
            panel,
            text="Starting...",
            fg=COLOR_TEXT_LIGHT,
            bg=COLOR_BACKGROUND,
            justify=tk.LEFT,
            anchor="w",
        )
        self.run_status.pack(fill=tk.X, pady=(0, 18))

        # Label that shows the final result (SUCCESS / FAILED) or live status.
        self.result_status = tk.Label(
            panel,
            text="",
            fg=COLOR_TEXT_LIGHT,
            bg=COLOR_BACKGROUND,
            justify=tk.LEFT,
            anchor="w",
        )
        self.result_status.pack(fill=tk.X, pady=(0, 18))

        # Buttons.
        self.pause_button = ttk.Button(
            panel,
            text="Pause",
            command=self.toggle_pause,
            style="Action.TButton",
        )
        self.pause_button.pack(fill=tk.X, pady=4)

        new_seed_button = ttk.Button(
            panel,
            text="New Seed",
            command=lambda: self.start_simulation(self.level),
            style="Action.TButton",
        )
        new_seed_button.pack(fill=tk.X, pady=4)

        restart_button = ttk.Button(
            panel,
            text="Restart",
            command=lambda: self.start_simulation(self.level, self.seed),
            style="Action.TButton",
        )
        restart_button.pack(fill=tk.X, pady=4)

        main_menu_button = ttk.Button(
            panel,
            text="Main Menu",
            command=self.show_main_menu,
            style="Action.TButton",
        )
        main_menu_button.pack(fill=tk.X, pady=4)

        # Colour legend at the bottom of the panel.
        legend = tk.Label(
            panel,
            text="Green: Pursuers\nRed: Evader\nGold: Goal\nBrown: Barrel\nBlue line: Wall",
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BACKGROUND,
            justify=tk.LEFT,
            anchor="w",
        )
        legend.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

    # Drawing the board

    def _cell_box(self, position: Position) -> tuple[float, float, float, float]:
        """
        Convert a (row, col) cell into pixel coordinates on the canvas.

        Returns (left, top, right, bottom). Uses the layout numbers computed
        at the start of _draw().
        """
        row, col = position

        left = self._offset_x + col * self._cell_size
        top = self._offset_y + row * self._cell_size
        right = left + self._cell_size
        bottom = top + self._cell_size

        return left, top, right, bottom

    def _draw_circle(
        self,
        position: Position,
        inset: float,
        fill: str,
        outline: str,
        width: int = 2,
    ) -> None:
        """
        Draw a circle inside a cell.

        `inset` is the fraction of the cell left empty on each side
        (0.2 means the circle is 60% of the cell width, centred).
        """
        left, top, right, bottom = self._cell_box(position)
        margin = self._cell_size * inset

        self.canvas.create_oval(
            left + margin,
            top + margin,
            right - margin,
            bottom - margin,
            fill=fill,
            outline=outline,
            width=width,
        )

    def _draw_grid(self) -> None:
        """Draw the empty background squares of the board."""
        environment = self.environment
        if environment is None:
            return
        scenario = environment.scenario

        for row in range(scenario.rows):
            for col in range(scenario.cols):
                self.canvas.create_rectangle(
                    *self._cell_box((row, col)),
                    fill="#1e293b",
                    outline="#334155",
                )

    def _draw_goals(self) -> None:
        """Draw each goal as a gold square."""
        environment = self.environment
        if environment is None:
            return
        for goal in environment.scenario.goals:
            self.canvas.create_rectangle(
                *self._cell_box(goal),
                fill="#d4a72c",
                outline="#f8d66d",
            )

    def _draw_barrels(self) -> None:
        """Draw each barrel as a brown circle."""
        environment = self.environment
        if environment is None:
            return
        for barrel in environment.scenario.barrels:
            self._draw_circle(barrel, inset=0.16, fill="#8b5e3c", outline="#d9a066")

    def _draw_walls(self) -> None:
        """
        Draw walls as blue lines along one edge of a cell.

        Each wall is stored as (cell, direction), e.g. ((2, 3), "up") is the
        wall on the top edge of the cell at row 2, column 3.
        """
        line_width = max(3, int(self._cell_size * 0.08))

        environment = self.environment
        if environment is None:
            return

        for position, direction in environment.scenario.walls:
            left, top, right, bottom = self._cell_box(position)

            # Line coordinates (x1, y1, x2, y2) for each possible edge.
            edges = {
                "up": (left, top, right, top),
                "down": (left, bottom, right, bottom),
                "left": (left, top, left, bottom),
                "right": (right, top, right, bottom),
            }

            self.canvas.create_line(
                *edges[direction],
                fill="#60a5fa",
                width=line_width,
            )

    def _draw_pursuers(self) -> None:
        """Draw each pursuer as a green circle."""
        environment = self.environment
        if environment is None:
            return
        for pursuer in environment.state.pursuers:
            self._draw_circle(pursuer, inset=0.20, fill="#39d98a", outline="#b7f7d0")

    def _draw_evaders(self) -> None:
        """Draw each evader as a red circle."""
        environment = self.environment
        if environment is None:
            return
        for evader in environment.state.evaders:
            self._draw_circle(evader, inset=0.18, fill="#f25f5c", outline="#ffd1d0")

    def _draw(self) -> None:
        """
        Redraw the entire board from scratch.

        Layers are drawn bottom to top so later layers appear on top:
        grid -> goals -> barrels -> walls -> pursuers -> evaders
        """
        if self.environment is None:
            return

        self.canvas.delete("all")
        environment = self.environment
        if environment is None:
            return
        scenario = environment.scenario

        # Work out the cell size so the whole board fits in the canvas
        # (leaving a 10 pixel margin on every side).
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        cell_size_from_width = (canvas_width - 20) / scenario.cols
        cell_size_from_height = (canvas_height - 20) / scenario.rows
        self._cell_size = min(cell_size_from_width, cell_size_from_height)

        # Centre the board inside the canvas.
        self._offset_x = (canvas_width - self._cell_size * scenario.cols) / 2
        self._offset_y = (canvas_height - self._cell_size * scenario.rows) / 2

        self._draw_grid()
        self._draw_goals()
        self._draw_barrels()
        self._draw_walls()
        self._draw_pursuers()
        self._draw_evaders()

    # The simulation loop
    def tick(self) -> None:
        """
        Runs once every SIMULATION_INTERVAL_MS milliseconds.

        Each tick: advance the game (unless paused/finished), redraw the board,
        refresh the side panel, and schedule the next tick.
        """
        # Stop the loop if the simulation was closed or never started.
        if not self.running or self.environment is None:
            return

        # 1. Advance the simulation by one step
        if not self.paused and not self.environment.done:
            pursuer_actions, evader_actions = _autonomous_actions(
                self.environment,
                self.algorithm,
            )
            self.environment.step(pursuer_actions, evader_actions)

            if self.environment.done:
                # Run just ended: freeze the clock at the final time.
                self.elapsed += perf_counter() - self.started_at
                self.started_at = 0.0
            else:
                # Still running: refresh the displayed time.
                self.elapsed = perf_counter() - self.started_at

        # 2. Redraw the board
        self._draw()

        # 3. Update the side panel
        self._update_status_panel()

        # 4. Schedule the next tick
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _update_status_panel(self) -> None:
        """Refresh the info text and the result text in the right-hand panel."""
        environment = self.environment
        if environment is None:
            return
        scenario = environment.scenario
        terminal_info = self._terminal_info()

        # Decide what the result label should say.
        if terminal_info:
            result = self._result_text(terminal_info)
        elif self.paused:
            result = "Paused"
        else:
            result = "Running autonomously"

        # Build the info text one line at a time.
        status_lines = [
            f"Mode: {MODE_LABELS[self.mode]}",
            f"Algorithm: {self.algorithm}",
            f"Seed: {self.seed}",
            f"Level {self.level} | Grid {scenario.rows} x {scenario.cols}",
            f"Pursuers: {len(scenario.pursuers)}",
            f"Objects: {len(scenario.barrels)} barrels, {len(scenario.walls)} walls",
            f"Time: {self.elapsed:.1f}s",
            f"Step: {environment.state.step_count}",
        ]
        self.run_status.config(text="\n".join(status_lines))

        # Green for success, red for failure, grey while still running.
        if terminal_info and terminal_info["reached_goal"]:
            result_color = COLOR_SUCCESS
        elif terminal_info:
            result_color = COLOR_FAILURE
        else:
            result_color = COLOR_TEXT_LIGHT

        self.result_status.config(text=result, fg=result_color)

    # Result handling
    def _terminal_info(self) -> dict | None:
        """
        Work out how the run ended.

        Returns None while the run is still going. Once finished, returns:
            {"captured": True/False, "reached_goal": True/False}
        """
        if self.environment is None or not self.environment.done:
            return None

        environment = self.environment
        if environment is None:
            return None

        state = environment.state
        scenario = environment.scenario

        # Captured: any evader is standing on a pursuer's cell.
        captured = False
        for evader in state.evaders:
            if evader in state.pursuers:
                captured = True

        # Reached goal: any evader is standing on a goal cell.
        reached_goal = False
        for evader in state.evaders:
            if evader in scenario.goals:
                reached_goal = True

        return {"captured": captured, "reached_goal": reached_goal}

    def _result_text(self, info: dict) -> str:
        """Turn the outcome flags into the message shown in the result label."""
        if info["reached_goal"]:
            return "SUCCESS\nEvader reached the goal"

        if info["captured"]:
            return "FAILED\nEvader was captured"

        return "FAILED\nEvader did not reach the goal"

    # Pause button
    def toggle_pause(self) -> None:
        """
        Pause or resume the simulation.

        When pausing, the time elapsed so far is saved. When resuming, the
        clock restarts from "now" so paused time is not counted.
        """
        if self.environment is None or self.environment.done:
            return

        now = perf_counter()
        self.paused = not self.paused

        if self.paused:
            # Save the time accumulated up to this moment.
            self.elapsed += now - self.started_at
            self.started_at = 0.0
            self.pause_button.config(text="Resume")
        else:
            # Start counting again from now.
            self.started_at = now
            self.pause_button.config(text="Pause")

    # Entry point
    def run(self) -> None:
        """Hand control to Tkinter. This call blocks until the window closes."""
        self.root.mainloop()


def run_gui(level: int = 1, seed: int = 7) -> None:
    """Create the app and start it. Call this from your main script."""
    app = PursuitEvasionApp(level, seed)
    app.run()