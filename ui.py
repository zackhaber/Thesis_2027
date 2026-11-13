"""
ui.py

Graphical User Interface and evaluation controller for Pursuit-Evasion scenarios.
Integrates Visual Rollouts, Batch Benchmarks, and Offline MARL Model Training.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from collections import deque
from dataclasses import replace
from pathlib import Path
from random import SystemRandom
from threading import Thread
from time import perf_counter
from typing import cast

from env import PursuitEvasionEnv, ScenarioSpec, generate_scenario
from env.env import Action
from env.game_modes import GameMode
from env.knowledge import (
    EvaderBelief,
    KnowledgeMode,
    Perception,
    PursuerBelief,
    explore_action,
    planner_view,
    target_action,
)
from evader_policy import evader_action
from marl.trainer import TorchMixer
from marl.base import AgentNetwork, belief_action_masks, belief_state_tensor, state_tensor
from planners import cbs_plan, mstar_plan, voronoi_plan
from search.viper_adapter import ViperSearcher

Position = tuple[int, int]
MOVEMENT_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")

MODE_LABELS = {
    GameMode.PURE_PURSUIT: "Pure Pursuit",
    GameMode.REACH_AVOID: "Reach and Avoid",
    GameMode.MAZE: "Maze",
}

CLASSICAL_ALGORITHMS = ("CBS", "M*", "Voronoi")
MARL_ALGORITHMS = ("IQL", "VDN", "QMIX", "QTRAN", "QPLEX")
ACTION_NAMES: tuple[Action, ...] = ("stay", "up", "down", "left", "right")

SIMULATION_INTERVAL_MS = 600
SEED_GENERATOR = SystemRandom()
MAX_SEED = 2**32 - 1

COLOR_BACKGROUND = "#111827"
COLOR_TEXT_MAIN = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_TEXT_LIGHT = "#cbd5e1"
COLOR_ERROR = "#fca5a5"
COLOR_SUCCESS = "#22c55e"
COLOR_FAILURE = "#f87171"
COLOR_ACCENT = "#2563eb"
DEFAULT_PURSUER_COUNT = 4
UNKNOWN_MAX_STEPS_MULTIPLIER = 4
_MARL_POLICY_CACHE: dict[Path, AgentNetwork] = {}
_MARL_PAYLOAD_CACHE: dict[Path, dict] = {}


# Board and Navigation Helpers

def _distance(first: Position, second: Position) -> int:
    """Calculates Manhattan distance between two grid coordinates."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _nearest_target(position: Position, targets: tuple[Position, ...]) -> Position:
    """Returns the target closest to position by Manhattan distance."""
    return min(targets, key=lambda target: _distance(position, target))


def _position_after_action(
    environment: PursuitEvasionEnv,
    position: Position,
    action: Action,
    speed: int = 1,
) -> Position:
    """Queries environment physics to get the resulting grid cell."""
    return environment._move(position, action, speed)


def _legal_actions(
    environment: PursuitEvasionEnv,
    position: Position,
) -> list[Action]:
    """Returns all movement actions resulting in a position change."""
    legal = []
    for action in MOVEMENT_ACTIONS:
        if _position_after_action(environment, position, action) != position:
            legal.append(action)
    return legal


def _first_step_to_target(
    environment: PursuitEvasionEnv,
    start: Position,
    target: Position,
    blocked_positions: set[Position] | None = None,
) -> Action | None:
    """Finds the shortest BFS route to a target cell and returns its first step action."""
    blocked = blocked_positions or set()
    queue: deque[tuple[Position, Action | None]] = deque([(start, None)])
    visited = {start}

    while queue:
        position, first_action = queue.popleft()
        if position == target:
            return first_action

        for action in MOVEMENT_ACTIONS:
            candidate = _position_after_action(environment, position, action, 1)
            if candidate == position or candidate in visited or candidate in blocked:
                continue

            visited.add(candidate)
            route_first_action = action if first_action is None else first_action
            queue.append((candidate, route_first_action))

    return None


def _fallback_movement_action(
    environment: PursuitEvasionEnv,
    position: Position,
    targets: tuple[Position, ...] = (),
) -> Action:
    """Selects a greedy fallback move when path search finds no route."""
    legal = _legal_actions(environment, position)
    if not legal:
        return MOVEMENT_ACTIONS[0]
    if not targets:
        return legal[0]

    return min(
        legal,
        key=lambda act: min(_distance(_position_after_action(environment, position, act), t) for t in targets),
    )


# Agent Action Selection

def _pursuer_action(environment: PursuitEvasionEnv, pursuer: Position) -> Action:
    """Computes pursuer step toward nearest evader via BFS or fallback."""
    evaders = tuple(environment.state.evaders)
    target = _nearest_target(pursuer, evaders)
    path_action = _first_step_to_target(environment, pursuer, target)
    return path_action if path_action is not None else _fallback_movement_action(environment, pursuer, (target,))


def _evader_action(
    environment: PursuitEvasionEnv,
    evader: Position,
    pursuer_positions: set[Position],
) -> Action:
    """Computes evasive movement through the shared policy implementation."""
    return evader_action(environment, evader, pursuer_positions)


def _planner_scenario(environment: PursuitEvasionEnv):
    """Creates scenario copy anchored to current agent positions."""
    return replace(
        environment.scenario,
        pursuers=tuple(environment.state.pursuers),
        evaders=tuple(environment.state.evaders),
    )


def _resolve_checkpoint_path(
    algorithm: str,
    training_seed: int,
    grid_size: int,
    pursuer_count: int,
    checkpoint_dir: Path = Path("checkpoints"),
    knowledge: KnowledgeMode = KnowledgeMode.KNOWN,
) -> Path | None:
    """Resolves the best available checkpoint for one MARL configuration."""
    del grid_size, pursuer_count
    knowledge_suffix = f"_{knowledge.value}" if knowledge == KnowledgeMode.UNKNOWN else ""
    stem = f"{algorithm.lower()}{knowledge_suffix}_{training_seed}"
    best = checkpoint_dir / f"{stem}_best.pt"
    plain = checkpoint_dir / f"{stem}.pt"
    return best if best.exists() else plain if plain.exists() else None


def _checkpoint_configuration_warning(
    checkpoint_path: Path,
    evaluation_mode: GameMode,
    wall_density: float,
    barrel_density: float,
    knowledge: KnowledgeMode = KnowledgeMode.KNOWN,
) -> str | None:
    """Describes evaluation settings that differ from checkpoint metadata."""
    payload = _load_marl_payload(checkpoint_path)
    differences: list[str] = []
    checkpoint_knowledge = payload.get("knowledge_mode", KnowledgeMode.KNOWN.value)
    if checkpoint_knowledge != knowledge.value:
        differences.append(f"knowledge {checkpoint_knowledge} vs {knowledge.value}")
    training_mode = payload.get("training_mode")
    if training_mode is None:
        differences.append("training mode is unavailable")
    elif training_mode != evaluation_mode.value:
        differences.append(f"mode {training_mode} vs {evaluation_mode.value}")
    evader_policy = payload.get("evader_policy")
    if evader_policy is None:
        differences.append("evader behaviour is unavailable")
    elif evader_policy != "scripted_flee":
        differences.append(f"evader {evader_policy} vs scripted_flee")
    trained_walls = payload.get("training_wall_density")
    trained_barrels = payload.get("training_barrel_density")
    if trained_walls is None or trained_barrels is None:
        differences.append("density settings are unavailable")
    else:
        if float(trained_walls) != wall_density:
            differences.append(f"walls {trained_walls} vs {wall_density}")
        if float(trained_barrels) != barrel_density:
            differences.append(f"barrels {trained_barrels} vs {barrel_density}")
    return (
        "Checkpoint training configuration differs: " + "; ".join(differences)
        if differences
        else None
    )


def _load_marl_payload(checkpoint_path: Path) -> dict:
    """Loads and caches checkpoint metadata."""
    if checkpoint_path not in _MARL_PAYLOAD_CACHE:
        import torch

        _MARL_PAYLOAD_CACHE[checkpoint_path] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    return _MARL_PAYLOAD_CACHE[checkpoint_path]


def _load_marl_policy(checkpoint_path: Path) -> tuple[AgentNetwork, dict]:
    """Loads and caches a MARL policy and its checkpoint metadata."""
    payload = _load_marl_payload(checkpoint_path)
    if checkpoint_path not in _MARL_POLICY_CACHE:
        policy = AgentNetwork(int(payload["state_size"]), int(payload["agent_count"]))
        policy.load_state_dict(payload["agents"])
        policy.eval()
        _MARL_POLICY_CACHE[checkpoint_path] = policy
    return _MARL_POLICY_CACHE[checkpoint_path], _MARL_PAYLOAD_CACHE[checkpoint_path]


def _marl_pursuer_actions(
    environment: PursuitEvasionEnv,
    algorithm: str,
    checkpoint_path: Path | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.KNOWN,
    perception: Perception | None = None,
) -> tuple[list[Action], bool, str | None]:
    """Returns MARL pursuer actions, status, and an optional checkpoint diagnostic."""
    if knowledge_mode == KnowledgeMode.UNKNOWN:
        if perception is None:
            return ["stay"] * len(environment.state.pursuers), False, "Unknown MARL evaluation requires a perception belief."
        fallback = [
            explore_action(perception.pursuer_belief, position)
            for position in perception.pursuer_belief.pursuer_positions
        ]
    else:
        fallback = [_pursuer_action(environment, p) for p in environment.state.pursuers]
    if checkpoint_path is None:
        return fallback, False, "No checkpoint was resolved for this configuration."

    import torch
    observation = environment.observe()
    try:
        payload = _load_marl_payload(checkpoint_path)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        return fallback, False, f"Checkpoint could not be loaded: {error}"
    if not isinstance(payload, dict):
        return fallback, False, "Checkpoint mismatch: payload is not a metadata dictionary."

    expected_state_size = (
        environment.scenario.rows * environment.scenario.cols * 9
        if knowledge_mode == KnowledgeMode.UNKNOWN
        else len(observation["global_state"])
    )
    expected_agent_count = len(observation["pursuers"])
    expected_grid_size = environment.scenario.rows
    metadata = (
        ("state_size", payload.get("state_size"), expected_state_size),
        ("agent_count", payload.get("agent_count"), expected_agent_count),
        ("grid_size", payload.get("grid_size"), expected_grid_size),
        ("knowledge_mode", payload.get("knowledge_mode", KnowledgeMode.KNOWN.value), knowledge_mode.value),
    )
    for name, actual, expected in metadata:
        if actual != expected:
            return (
                fallback,
                False,
                f"Checkpoint mismatch: {name} is {actual}, expected {expected}.",
            )
    try:
        policy, _ = _load_marl_policy(checkpoint_path)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        return fallback, False, f"Checkpoint could not be loaded: {error}"

    if knowledge_mode == KnowledgeMode.UNKNOWN:
        state = belief_state_tensor(perception.pursuer_belief, environment)
        masks = torch.tensor(
            belief_action_masks(perception.pursuer_belief, environment),
            dtype=torch.bool,
        )
    else:
        state = state_tensor(observation)
        masks = torch.tensor(observation["action_masks"]["pursuers"], dtype=torch.bool)
    with torch.no_grad():
        values = policy(
            state.repeat(expected_agent_count, 1),
            torch.arange(expected_agent_count),
        )

    actions: list[Action] = []
    for index, mask in enumerate(masks):
        masked = values[index].clone()
        masked[~mask] = -torch.inf
        actions.append(ACTION_NAMES[int(masked.argmax())])
    return actions, True, None


def _planner_action(value: object, fallback: Action) -> Action:
    if isinstance(value, str) and value in ACTION_NAMES:
        return cast(Action, value)
    return fallback


def _autonomous_actions(
    environment: PursuitEvasionEnv,
    algorithm: str = "CBS",
    checkpoint_path: Path | None = None,
    knowledge_mode: KnowledgeMode = KnowledgeMode.KNOWN,
    perception: Perception | None = None,
    viper_searcher: ViperSearcher | None = None,
    switch_on_detection: bool = True,
) -> tuple[list[Action], list[Action], bool, str | None]:
    """Computes coordinated timestep moves for all agents."""
    if knowledge_mode == KnowledgeMode.UNKNOWN:
        if perception is None:
            perception = Perception()
        if viper_searcher is not None:
            sensor = replace(perception.sensor, radius=viper_searcher.sensor_radius_cells)
            perception.sensor = sensor
            perception.pursuer_belief.sensor = sensor
            perception.evader_belief.sensor = sensor
        perception.update(environment)
        return _unknown_actions(
            environment,
            algorithm,
            perception,
            checkpoint_path,
            viper_searcher,
            switch_on_detection,
        )

    is_trained = True
    checkpoint_message: str | None = None
    pursuer_actions: list[Action]
    if algorithm in {"CBS", "M*", "Voronoi"}:
        planner = {"CBS": cbs_plan, "M*": mstar_plan, "Voronoi": voronoi_plan}[algorithm]
        result = planner(_planner_scenario(environment))
        pursuer_actions = []
        for index, pursuer in enumerate(environment.state.pursuers):
            paths = result["paths"]
            path = paths[index] if index < len(paths) else []
            pursuer_actions.append(
                _planner_action(path[0] if path else None, _pursuer_action(environment, pursuer))
            )
    else:
        pursuer_actions, is_trained, checkpoint_message = _marl_pursuer_actions(
            environment, algorithm, checkpoint_path
        )

    predicted_positions: set[Position] = set()
    for pursuer, action in zip(environment.state.pursuers, pursuer_actions):
        next_pos = _position_after_action(environment, pursuer, action, environment.scenario.pursuer_speed)
        predicted_positions.add(pursuer)
        predicted_positions.add(next_pos)

    evader_actions: list[Action] = [
        _evader_action(environment, evader, predicted_positions)
        for evader in environment.state.evaders
    ]

    return pursuer_actions, evader_actions, is_trained, checkpoint_message


def _unknown_actions(
    environment: PursuitEvasionEnv,
    algorithm: str,
    perception: Perception,
    checkpoint_path: Path | None = None,
    viper_searcher: ViperSearcher | None = None,
    switch_on_detection: bool = True,
) -> tuple[list[Action], list[Action], bool, str | None]:
    """Selects actions from partial beliefs without changing Known behavior."""
    pursuer_belief = perception.pursuer_belief
    evader_belief = perception.evader_belief
    if algorithm in MARL_ALGORITHMS:
        pursuer_actions, is_trained, checkpoint_message = _marl_pursuer_actions(
            environment,
            algorithm,
            checkpoint_path,
            KnowledgeMode.UNKNOWN,
            perception,
        )
    elif (
        viper_searcher is not None
        and (pursuer_belief.visible_evaders or pursuer_belief.last_seen_evader)
    ):
        viper_searcher.selected_algorithm = algorithm
        viper_searcher.switch_on_detection = switch_on_detection
        viper_searcher.set_environment(environment)
        pursuer_actions = viper_searcher.act(
            pursuer_belief,
            pursuer_belief.pursuer_positions,
        )
    elif pursuer_belief.visible_evaders or pursuer_belief.last_seen_evader:
        view = planner_view(pursuer_belief, environment)
        planner = {"CBS": cbs_plan, "M*": mstar_plan, "Voronoi": voronoi_plan}[algorithm]
        result = planner(view)
        pursuer_actions = []
        for index, position in enumerate(pursuer_belief.pursuer_positions):
            paths = result["paths"]
            path = paths[index] if index < len(paths) else []
            pursuer_actions.append(
                _planner_action(path[0], explore_action(pursuer_belief, position))
                if path
                else explore_action(pursuer_belief, position)
            )
    else:
        pursuer_actions = [
            explore_action(pursuer_belief, position)
            for position in pursuer_belief.pursuer_positions
        ]

    evader_actions = []
    for position in environment.state.evaders:
        if evader_belief.known_goals:
            goal = min(
                evader_belief.known_goals,
                key=lambda candidate: abs(position[0] - candidate[0]) + abs(position[1] - candidate[1]),
            )
            evader_actions.append(
                target_action(evader_belief, position, goal, evader_belief.visible_pursuers)
            )
        else:
            evader_actions.append(
                explore_action(
                    evader_belief,
                    position,
                    evader_belief.visible_pursuers,
                )
            )
    if algorithm in MARL_ALGORITHMS:
        return pursuer_actions, evader_actions, is_trained, checkpoint_message
    return pursuer_actions, evader_actions, False, None


# Application Interface
class PursuitEvasionApp:
    """Main application supporting visual rollouts, batch benchmarks, and MARL training."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.mode = GameMode.PURE_PURSUIT
        self.algorithm = "CBS"
        self.custom_grid_size = 20
        self.custom_wall_density = 0.5
        self.custom_barrel_density = 0.1
        self.knowledge = KnowledgeMode.KNOWN
        self.switch_on_detection = True
        self.viper_searcher = ViperSearcher(selected_algorithm=self.algorithm)
        self.show_ground_truth = False

        self.environment: PursuitEvasionEnv | None = None
        self.perception: Perception | None = None
        self.detection_step: int | None = None
        self.checkpoint_path: Path | None = None
        self.checkpoint_message: str | None = None
        self.running = False
        self.paused = False
        self.is_checkpoint_active = False
        self._simulation_generation = 0

        self.started_at = 0.0
        self.elapsed = 0.0

        self.root = tk.Tk()
        self._configure_styles()
        self.root.title("Multi-Agent Pursuit-Evasion Platform")
        self.root.configure(bg=COLOR_BACKGROUND)
        self.root.minsize(950, 700)

        self.show_main_menu()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Action.TButton",
            anchor="center",
            justify="center",
            font=("Segoe UI", 11, "bold"),
            padding=(18, 11),
            foreground=COLOR_TEXT_MAIN,
            background=COLOR_ACCENT,
            borderwidth=0,
        )
        style.map(
            "Action.TButton",
            background=[("active", "#3b82f6"), ("pressed", "#1d4ed8")],
            foreground=[("disabled", COLOR_TEXT_MUTED)],
        )

    def _clear(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def _frame(self) -> tk.Frame:
        frame = tk.Frame(self.root, bg=COLOR_BACKGROUND)
        frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        return frame

    def _title(self, parent: tk.Misc, text: str, subtitle: str) -> None:
        tk.Label(parent, text=text, fg=COLOR_TEXT_MAIN, bg=COLOR_BACKGROUND, font=("Segoe UI", 24, "bold")).pack(pady=(15, 6))
        tk.Label(parent, text=subtitle, fg=COLOR_TEXT_MUTED, bg=COLOR_BACKGROUND, font=("Segoe UI", 11)).pack(pady=(0, 20))

    def _button(self, parent: tk.Misc, text: str, command, width: int = 28) -> None:
        ttk.Button(parent, text=text, command=command, width=width, style="Action.TButton").pack(pady=6)

    def _set_random_seed(self, entry: ttk.Entry, on_change=None) -> None:
        """Replace a seed entry with a valid random seed and refresh dependent UI."""
        entry.delete(0, tk.END)
        entry.insert(0, str(SEED_GENERATOR.randint(0, MAX_SEED)))
        if on_change is not None:
            on_change()

    # Main Menu Navigation
    def show_main_menu(self) -> None:
        """Renders main navigation hub."""
        self.running = False
        self._simulation_generation += 1
        self._clear()
        frame = self._frame()
        self._title(frame, "Multi-Agent Strategies for\n Pursuit-Evasion Games", "By Zack Haber")
        self._button(frame, "Interactive Simulation", self.show_scenario_config)
        self._button(frame, "Algorithm Evaluation", self.show_simulation_config) # Evaluates classical planners and MARL models across multiple seeds
        self._button(frame, "MARL Training Panel", self.show_marl_training_config)
        self._button(frame, "Quit Application", self.root.destroy)

    def _configuration_form(self, frame: tk.Frame, simulation: bool = False) -> tk.Frame:
        """Constructs configuration input form with live checkpoint detection."""
        form = tk.Frame(frame, bg=COLOR_BACKGROUND)
        form.pack(pady=8)

        def add_label(row: int, text: str) -> None:
            tk.Label(form, text=text, fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, font=("Segoe UI", 11)).grid(row=row, column=0, padx=10, pady=5, sticky="e")

        add_label(0, "Game Mode")
        self.config_mode = ttk.Combobox(form, values=("Maze", "Reach and Avoid", "Pure Pursuit"), state="readonly", width=25)
        self.config_mode.set("Reach and Avoid")
        self.config_mode.grid(row=0, column=1, padx=10, pady=5)

        add_label(1, "Algorithm Family")
        self.config_family = ttk.Combobox(form, values=("Planner", "MARL"), state="readonly", width=25)
        self.config_family.set("Planner")
        self.config_family.grid(row=1, column=1, padx=10, pady=5)
        self.config_family.bind("<<ComboboxSelected>>", lambda *_: self._update_config_algorithms())

        add_label(7, "Environment Knowledge")
        self.config_knowledge = ttk.Combobox(
            form, values=("Known", "Unknown (Search)"), state="readonly", width=25
        )
        self.config_knowledge.set("Known")
        self.config_knowledge.grid(row=7, column=1, padx=10, pady=5)
        self.config_knowledge.bind("<<ComboboxSelected>>", lambda *_: self._update_config_algorithms())

        add_label(2, "Algorithm")
        self.config_algorithm = ttk.Combobox(form, values=CLASSICAL_ALGORITHMS, state="readonly", width=25)
        self.config_algorithm.set(CLASSICAL_ALGORITHMS[0])
        self.config_algorithm.grid(row=2, column=1, padx=10, pady=5)
        self.config_algorithm.bind("<<ComboboxSelected>>", lambda *_: self._check_checkpoint_status())

        add_label(3, "Scenario Seed")
        seed_controls = tk.Frame(form, bg=COLOR_BACKGROUND)
        seed_controls.grid(row=3, column=1, padx=10, pady=5)
        self.config_seed_entry = ttk.Entry(seed_controls, width=19, justify="center")
        self.config_seed_entry.insert(0, str(self.seed))
        self.config_seed_entry.pack(side=tk.LEFT)
        ttk.Button(
            seed_controls,
            text="Random",
            command=lambda: self._set_random_seed(
                self.config_seed_entry, self._check_checkpoint_status
            ),
            width=8,
        ).pack(side=tk.LEFT, padx=(6, 0))
        self.config_seed_entry.bind("<KeyRelease>", lambda *_: self._check_checkpoint_status())

        add_label(4, "Grid Size (X = Y)")
        self.config_size = ttk.Entry(form, width=28, justify="center")
        self.config_size.insert(0, "20")
        self.config_size.grid(row=4, column=1, padx=10, pady=5)

        add_label(5, "Walls Density (0.0 - 1.0)")
        self.config_walls = ttk.Spinbox(form, from_=0.0, to=1.0, increment=0.1, width=25)
        self.config_walls.set("0.5")
        self.config_walls.grid(row=5, column=1, padx=10, pady=5)

        add_label(6, "Barrels Density (0.0 - 1.0)")
        self.config_barrels = ttk.Spinbox(form, from_=0.0, to=1.0, increment=0.1, width=25)
        self.config_barrels.set("0.1")
        self.config_barrels.grid(row=6, column=1, padx=10, pady=5)

        if simulation:
            add_label(8, "Number of Simulations")
            self.batch_runs = ttk.Entry(form, width=28, justify="center")
            self.batch_runs.insert(0, "10")
            self.batch_runs.grid(row=8, column=1, padx=10, pady=5)

        add_label(9, "Switch to selected algorithm on detection")
        self.config_switch_detection = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            form,
            variable=self.config_switch_detection,
        ).grid(row=9, column=1, padx=10, pady=5, sticky="w")

        self.checkpoint_status_label = tk.Label(frame, text="", fg=COLOR_TEXT_MUTED, bg=COLOR_BACKGROUND, font=("Segoe UI", 10))
        self.checkpoint_status_label.pack(pady=4)

        self.config_error = tk.Label(frame, text="", fg=COLOR_ERROR, bg=COLOR_BACKGROUND, font=("Segoe UI", 10))
        self.config_error.pack(pady=4)
        return form

    def _update_config_algorithms(self) -> None:
        unknown = self.config_knowledge.get() == "Unknown (Search)"
        seed = int(self.config_seed_entry.get().strip()) if self.config_seed_entry.get().strip().lstrip("-").isdigit() else self.seed
        grid_size = int(self.config_size.get().strip()) if self.config_size.get().strip().isdigit() else self.custom_grid_size
        knowledge = KnowledgeMode.UNKNOWN if unknown else KnowledgeMode.KNOWN
        unknown_marl_available = any(
            _resolve_checkpoint_path(
                algorithm, seed, grid_size, DEFAULT_PURSUER_COUNT, knowledge=knowledge
            )
            for algorithm in MARL_ALGORITHMS
        )
        family_values = ("Planner", "MARL") if not unknown or unknown_marl_available else ("Planner",)
        self.config_family.configure(values=family_values)
        if self.config_family.get() not in family_values:
            self.config_family.set("Planner")
        values = (
            CLASSICAL_ALGORITHMS
            if self.config_family.get() == "Planner"
            else MARL_ALGORITHMS
        )
        self.config_algorithm.configure(values=values)
        self.config_algorithm.set(values[0])
        self._check_checkpoint_status()

    def _check_checkpoint_status(self) -> None:
        if self.config_family.get() == "Planner":
            self.checkpoint_status_label.config(
                text=(
                    "Info: Unknown mode uses sensor-limited planner search."
                    if self.config_knowledge.get() == "Unknown (Search)"
                    else "Info: Classical Planners execute exact search paths dynamically."
                ),
                fg=COLOR_TEXT_LIGHT,
            )
            return

        try:
            seed_val = int(self.config_seed_entry.get().strip())
        except ValueError:
            seed_val = self.seed
        try:
            grid_size = int(self.config_size.get().strip())
        except ValueError:
            grid_size = self.custom_grid_size

        ckpt_path = _resolve_checkpoint_path(
            self.config_algorithm.get(),
            seed_val,
            grid_size,
            DEFAULT_PURSUER_COUNT,
            knowledge=(
                KnowledgeMode.UNKNOWN
                if self.config_knowledge.get() == "Unknown (Search)"
                else KnowledgeMode.KNOWN
            ),
        )
        if ckpt_path is not None:
            try:
                evaluation_mode = {
                    "Maze": GameMode.MAZE,
                    "Reach and Avoid": GameMode.REACH_AVOID,
                    "Pure Pursuit": GameMode.PURE_PURSUIT,
                }.get(self.config_mode.get(), GameMode.PURE_PURSUIT)
                warning = _checkpoint_configuration_warning(
                    ckpt_path,
                    evaluation_mode,
                    float(self.config_walls.get()),
                    float(self.config_barrels.get()),
                    KnowledgeMode.UNKNOWN
                    if self.config_knowledge.get() == "Unknown (Search)"
                    else KnowledgeMode.KNOWN,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                warning = f"Checkpoint metadata could not be read: {error}"
            if warning:
                self.checkpoint_status_label.config(
                    text=f"Checkpoint Found: {ckpt_path}\nWarning: {warning}",
                    fg=COLOR_FAILURE,
                )
            else:
                self.checkpoint_status_label.config(
                    text=f"Checkpoint Found: {ckpt_path}", fg=COLOR_SUCCESS
                )
        else:
            knowledge_suffix = (
                "_unknown"
                if self.config_knowledge.get() == "Unknown (Search)"
                else ""
            )
            expected = Path("checkpoints") / f"{self.config_algorithm.get().lower()}{knowledge_suffix}_{seed_val}[_best].pt"
            self.checkpoint_status_label.config(text=f"Warning: No Checkpoint Found ({expected}). Will use fallback policy.", fg=COLOR_FAILURE)

    def _read_configuration(self) -> bool:
        try:
            size = int(self.config_size.get().strip())
            walls = float(self.config_walls.get())
            barrels = float(self.config_barrels.get())
            seed_val = int(self.config_seed_entry.get().strip())
        except ValueError:
            self.config_error.config(text="Enter valid numeric configuration values")
            return False

        if size < 5 or not 0 <= walls <= 1 or not 0 <= barrels <= 1:
            self.config_error.config(text="Size must be >= 5 and densities between 0.0 and 1.0")
            return False

        self.mode = {"Maze": GameMode.MAZE, "Reach and Avoid": GameMode.REACH_AVOID, "Pure Pursuit": GameMode.PURE_PURSUIT}[self.config_mode.get()]
        self.knowledge = (
            KnowledgeMode.UNKNOWN
            if self.config_knowledge.get() == "Unknown (Search)"
            else KnowledgeMode.KNOWN
        )
        self.algorithm = self.config_algorithm.get()
        self.switch_on_detection = bool(self.config_switch_detection.get())
        self.viper_searcher.selected_algorithm = self.algorithm
        self.viper_searcher.switch_on_detection = self.switch_on_detection
        self.seed = seed_val
        self.custom_grid_size = size
        self.custom_wall_density = walls
        self.custom_barrel_density = barrels
        self.checkpoint_path = (
            _resolve_checkpoint_path(
                self.algorithm,
                self.seed,
                self.custom_grid_size,
                DEFAULT_PURSUER_COUNT,
                knowledge=self.knowledge,
            )
            if self.algorithm in MARL_ALGORITHMS
            else None
        )
        self.evaluation_warning = None
        if self.checkpoint_path is not None:
            self.evaluation_warning = _checkpoint_configuration_warning(
                self.checkpoint_path,
                self.mode,
                self.custom_wall_density,
                self.custom_barrel_density,
                self.knowledge,
            )
        self.checkpoint_message = None
        self.perception = None
        self.detection_step = None
        self.viper_searcher.reset_episode()
        return True

    # Interactive Simulations

    def show_scenario_config(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Interactive Simulation Configuration", "Configure live 2D step-by-step playback parameters.")
        self._configuration_form(frame, simulation=False)
        self._check_checkpoint_status()
        self._button(frame, "Launch Simulation", self._start_configured_scenario)
        self._button(frame, "Main Menu", self.show_main_menu)

    def _start_configured_scenario(self) -> None:
        if self._read_configuration():
            self.start_simulation(self.seed)

    # Algorithm Evaluations

    def show_simulation_config(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Algorithm Evaluation Configuration", "Run headless evaluations across seeds to calculate performance metrics.")
        self._configuration_form(frame, simulation=True)
        self._check_checkpoint_status()
        self._button(frame, "Run  Evaluation", self._start_batch_simulation)
        self._button(frame, "Main Menu", self.show_main_menu)

    def _start_batch_simulation(self) -> None:
        try:
            runs = int(self.batch_runs.get().strip())
        except ValueError:
            self.config_error.config(text="Enter a valid whole number of simulation runs")
            return

        if not 1 <= runs <= 10000:
            self.config_error.config(text="Enter between 1 and 10,000 simulations")
            return

        if not self._read_configuration():
            return

        if self.algorithm in MARL_ALGORITHMS and self.checkpoint_path is None:
            proceed = messagebox.askyesno(
                "No MARL checkpoint",
                "No checkpoint matches this MARL configuration. Run the batch with the BFS fallback policy?",
            )
            if not proceed:
                return

        self._clear()
        frame = self._frame()
        self._title(frame, "Running Batch Benchmark", f"Evaluating {self.algorithm} across {runs} episode seed(s)...")
        self.batch_progress = tk.Label(frame, text="Calculating success rate, latency, and path efficiency...", fg=COLOR_TEXT_MUTED, bg=COLOR_BACKGROUND, font=("Segoe UI", 11))
        self.batch_progress.pack(pady=20)
        Thread(target=self._run_batch_worker, args=(runs,), daemon=True).start()

    def _run_batch_worker(self, runs: int) -> None:
        try:
            base_seed = self.seed
            captured = 0
            trained_steps = 0
            total_steps = 0
            explored_percentages: list[float] = []
            detection_steps: list[int] = []
            detected_episodes = 0
            capture_steps: list[int] = []
            planning_seconds: list[float] = []
            episode_seconds: list[float] = []

            for index in range(runs):
                episode_seed = (base_seed + index) % (2**32)
                scenario = generate_scenario(
                    ScenarioSpec(grid_size=self.custom_grid_size, wall_density=self.custom_wall_density, barrel_density=self.custom_barrel_density, seed=episode_seed),
                    self.mode,
                )
                environment = PursuitEvasionEnv(
                    scenario,
                    self.mode,
                    max_steps=self._max_steps_for_mode(scenario.rows, scenario.cols),
                    seed=episode_seed,
                )
                environment.reset()
                if self.knowledge == KnowledgeMode.UNKNOWN and hasattr(self, "viper_searcher"):
                    self.viper_searcher.reset_episode()
                perception = Perception() if self.knowledge == KnowledgeMode.UNKNOWN else None
                episode_detection_step: int | None = 0 if self.knowledge == KnowledgeMode.KNOWN else None
                episode_started = perf_counter()
                info: dict = {"captured": False, "time": 0}

                while not environment.done:
                    planning_started = perf_counter()
                    if self.knowledge == KnowledgeMode.UNKNOWN:
                        pursuer_actions, evader_actions, is_trained, checkpoint_message = _autonomous_actions(
                            environment,
                            self.algorithm,
                            self.checkpoint_path,
                            self.knowledge,
                            perception,
                            self.viper_searcher,
                            self.switch_on_detection,
                        )
                    else:
                        pursuer_actions, evader_actions, is_trained, checkpoint_message = _autonomous_actions(
                            environment,
                            self.algorithm,
                            self.checkpoint_path,
                            self.knowledge,
                            perception,
                        )
                    self.checkpoint_message = checkpoint_message
                    trained_steps += int(is_trained)
                    total_steps += 1
                    if (
                        self.knowledge == KnowledgeMode.UNKNOWN
                        and perception is not None
                        and perception.pursuer_belief.visible_evaders
                        and episode_detection_step is None
                    ):
                        episode_detection_step = environment.state.step_count
                    planning_seconds.append(perf_counter() - planning_started)
                    _, _, _, _, info = environment.step(pursuer_actions, evader_actions)

                if self.knowledge == KnowledgeMode.UNKNOWN and perception is not None:
                    explored_percentages.append(perception.pursuer_belief.coverage())
                    if episode_detection_step is not None:
                        detection_steps.append(episode_detection_step)
                        detected_episodes += 1
                else:
                    explored_percentages.append(100.0)
                    detection_steps.append(0)
                    detected_episodes += 1
                episode_seconds.append(perf_counter() - episode_started)
                if bool(info.get("captured")):
                    captured += 1
                    capture_steps.append(int(info.get("time", 0)))

            results = (
                captured / runs * 100,
                sum(capture_steps) / len(capture_steps) if capture_steps else 0.0,
                sum(planning_seconds) / len(planning_seconds) * 1000,
                sum(episode_seconds) / len(episode_seconds) * 1000,
                trained_steps / total_steps * 100 if total_steps else 0.0,
                sum(explored_percentages) / len(explored_percentages),
                sum(detection_steps) / len(detection_steps) if detection_steps else 0.0,
                detected_episodes / runs * 100,
            )
            self.root.after(0, lambda: self._show_batch_results(results, runs))
        except Exception as error:
            self.root.after(0, lambda: self._show_batch_error(error))

    def _show_batch_results(self, results: tuple[float, ...], runs: int) -> None:
        (
            success_rate,
            average_steps,
            latency_ms,
            average_time_ms,
            trained_percentage,
            explored_percentage,
            detection_steps,
            detection_rate,
        ) = results
        self._clear()
        frame = self._frame()
        self._title(
            frame,
            "Benchmark Results Summary",
            f"{self.algorithm} ({runs} Runs) | Mode: {MODE_LABELS[self.mode]} | "
            f"Knowledge: {self.knowledge.value}",
        )
        result_text = (
            f"• Success Rate (%): {success_rate:.1f}%\n\n"
            f"• Path Efficiency (Avg Steps to Capture): {average_steps:.1f}\n\n"
            f"• Latency / Decision Time per Step: {latency_ms:.3f} ms\n\n"
            f"• Total Average Episode Execution Time: {average_time_ms:.3f} ms\n\n"
            f"• Trained policy used in {trained_percentage:.1f}% of steps\n\n"
            f"• Explored Area (Avg): {explored_percentage:.1f}%\n\n"
            f"• Time to First Detection (Avg Steps): {detection_steps:.1f}\n\n"
            f"• Detection Rate: {detection_rate:.1f}%"
        )
        tk.Label(frame, text=result_text, fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, justify=tk.LEFT, font=("Segoe UI", 12)).pack(pady=20)
        if trained_percentage < 100:
            tk.Label(
                frame,
                text="Warning: The trained policy was not used for every step.",
                fg=COLOR_FAILURE,
                bg=COLOR_BACKGROUND,
                font=("Segoe UI", 11),
            ).pack()
        if getattr(self, "evaluation_warning", None):
            tk.Label(
                frame,
                text=f"Warning: {self.evaluation_warning}",
                fg=COLOR_FAILURE,
                bg=COLOR_BACKGROUND,
                font=("Segoe UI", 11),
                wraplength=700,
                justify=tk.LEFT,
            ).pack()
        self._button(frame, "Run New Benchmark", self.show_simulation_config)
        self._button(frame, "Main Menu", self.show_main_menu)

    def _show_batch_error(self, error: Exception) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Benchmark Execution Error", "An error occurred during evaluation.")
        tk.Label(frame, text=str(error), fg=COLOR_ERROR, bg=COLOR_BACKGROUND, wraplength=650, justify=tk.LEFT, font=("Segoe UI", 11)).pack(pady=20)
        self._button(frame, "Try Again", self.show_simulation_config)
        self._button(frame, "Main Menu", self.show_main_menu)

    # MARL Training Panel
    def show_marl_training_config(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "MARL Offline Training Panel", "Train multi-agent neural network models and export weight checkpoints.")

        form = tk.Frame(frame, bg=COLOR_BACKGROUND)
        form.pack(pady=8)

        def add_label(row: int, text: str) -> None:
            tk.Label(form, text=text, fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, font=("Segoe UI", 11)).grid(row=row, column=0, padx=10, pady=5, sticky="e")

        add_label(0, "MARL Algorithm")
        self.train_alg = ttk.Combobox(form, values=MARL_ALGORITHMS, state="readonly", width=25)
        self.train_alg.set("QMIX")
        self.train_alg.grid(row=0, column=1, padx=10, pady=5)

        add_label(1, "Training Episodes")
        self.train_episodes = ttk.Entry(form, width=28, justify="center")
        self.train_episodes.insert(0, "100")
        self.train_episodes.grid(row=1, column=1, padx=10, pady=5)

        add_label(2, "Scenario Seed")
        seed_controls = tk.Frame(form, bg=COLOR_BACKGROUND)
        seed_controls.grid(row=2, column=1, padx=10, pady=5)
        self.train_seed = ttk.Entry(seed_controls, width=19, justify="center")
        self.train_seed.insert(0, "0")
        self.train_seed.pack(side=tk.LEFT)
        ttk.Button(
            seed_controls,
            text="Random",
            command=lambda: self._set_random_seed(self.train_seed),
            width=8,
        ).pack(side=tk.LEFT, padx=(6, 0))

        add_label(3, "Grid Size")
        self.train_size = ttk.Entry(form, width=28, justify="center")
        self.train_size.insert(0, "20")
        self.train_size.grid(row=3, column=1, padx=10, pady=5)

        add_label(4, "Pursuers Count")
        self.train_pursuers = ttk.Entry(form, width=28, justify="center")
        self.train_pursuers.insert(0, "4")
        self.train_pursuers.grid(row=4, column=1, padx=10, pady=5)

        add_label(5, "Game Mode")
        self.train_mode = ttk.Combobox(
            form,
            values=("Maze", "Reach and Avoid", "Pure Pursuit"),
            state="readonly",
            width=25,
        )
        self.train_mode.set("Pure Pursuit")
        self.train_mode.grid(row=5, column=1, padx=10, pady=5)

        add_label(6, "Evader Behaviour")
        self.train_evader_policy = ttk.Combobox(
            form,
            values=("Static", "Scripted Flee"),
            state="readonly",
            width=25,
        )
        self.train_evader_policy.set("Static")
        self.train_evader_policy.grid(row=6, column=1, padx=10, pady=5)

        add_label(7, "Density Caps")
        self.train_cap_densities = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, variable=self.train_cap_densities).grid(
            row=7, column=1, padx=10, pady=5, sticky="w"
        )

        add_label(8, "Environment Knowledge")
        self.train_knowledge = ttk.Combobox(
            form,
            values=("Known", "Unknown (Search)"),
            state="readonly",
            width=25,
        )
        self.train_knowledge.set("Known")
        self.train_knowledge.grid(row=8, column=1, padx=10, pady=5)

        self.train_error = tk.Label(frame, text="", fg=COLOR_ERROR, bg=COLOR_BACKGROUND, font=("Segoe UI", 10))
        self.train_error.pack(pady=4)

        self._button(frame, "Start Model Training", self._start_marl_training)
        self._button(frame, "Main Menu", self.show_main_menu)

    def _start_marl_training(self) -> None:
        try:
            alg = self.train_alg.get()
            episodes = int(self.train_episodes.get().strip())
            seed = int(self.train_seed.get().strip())
            grid_size = int(self.train_size.get().strip())
            pursuers = int(self.train_pursuers.get().strip())
        except ValueError:
            self.train_error.config(text="Enter valid numeric parameters for training setup")
            return

        self._clear()
        frame = self._frame()
        self._title(frame, "Training MARL Model...", f"{alg} | Seed {seed} | {episodes} Episodes")

        self.train_progress_label = tk.Label(frame, text="Initializing training pipeline...", fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, font=("Segoe UI", 11))
        self.train_progress_label.pack(pady=10)

        log_frame = tk.Frame(frame, bg=COLOR_BACKGROUND)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.train_log_box = tk.Text(
            log_frame,
            height=24,
            width=115,
            wrap=tk.NONE,
            bg="#0f172a",
            fg=COLOR_TEXT_LIGHT,
            insertbackground=COLOR_TEXT_LIGHT,
            font=("Consolas", 10),
        )
        self.train_log_box.grid(row=0, column=0, sticky="nsew")

        vertical_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.train_log_box.yview
        )
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.HORIZONTAL, command=self.train_log_box.xview
        )
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.train_log_box.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

        Thread(
            target=self._run_training_worker,
            args=(
                alg,
                seed,
                episodes,
                grid_size,
                pursuers,
                {
                    "Maze": GameMode.MAZE,
                    "Reach and Avoid": GameMode.REACH_AVOID,
                    "Pure Pursuit": GameMode.PURE_PURSUIT,
                }[self.train_mode.get()],
                "scripted_flee"
                if self.train_evader_policy.get() == "Scripted Flee"
                else "static",
                bool(self.train_cap_densities.get()),
                KnowledgeMode.UNKNOWN
                if self.train_knowledge.get() == "Unknown (Search)"
                else KnowledgeMode.KNOWN,
            ),
            daemon=True,
        ).start()

    def _run_training_worker(
        self,
        alg: str,
        seed: int,
        episodes: int,
        grid_size: int,
        pursuers: int,
        training_mode: GameMode = GameMode.PURE_PURSUIT,
        evader_policy: str = "static",
        cap_densities: bool = True,
        knowledge: KnowledgeMode = KnowledgeMode.KNOWN,
    ) -> None:
        try:
            trainer = TorchMixer(
                algorithm=alg,
                seed=seed,
                checkpoint_dir="checkpoints",
                grid_size=grid_size,
                pursuers=pursuers,
                training_mode=training_mode,
                evader_policy=evader_policy,
                cap_densities=cap_densities,
                knowledge=knowledge,
            )

            def callback(
                ep: int,
                total_ep: int,
                reward: float,
                loss: float,
                captured: bool,
                steps: int,
                reward_average: float,
                loss_average: float,
            ):
                result = "Yes" if captured else "No"
                msg = (
                    f"Episode {ep}/{total_ep} | Reward: {reward:.2f} | "
                    f"TD Loss: {loss:.4f} | Captured: {result} | Steps: {steps} | "
                    f"25-Ep Avg Reward: {reward_average:.2f} | "
                    f"25-Ep Avg Loss: {loss_average:.4f}\n"
                )
                self.root.after(0, lambda: self._update_train_progress(ep, total_ep, msg))

            result = trainer.train(episodes=episodes, callback=callback)
            self.root.after(0, lambda: self._show_training_complete(result))
        except Exception as error:
            self.root.after(0, lambda: self._show_train_error(error))

    def _update_train_progress(self, ep: int, total_ep: int, log_msg: str) -> None:
        self.train_progress_label.config(text=f"Progress: Episode {ep} of {total_ep} completed")
        self.train_log_box.insert(tk.END, log_msg)
        self.train_log_box.see(tk.END)

    def _show_training_complete(self, result) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Training Completed!", f"Successfully exported model checkpoint for {result.algorithm}")
        info_text = (
            f"Checkpoint File: {result.checkpoint}\n"
            f"Final Avg Reward: {sum(result.rewards[-10:]) / max(1, len(result.rewards[-10:])):.2f}\n"
            f"Final Avg Loss: {sum(result.losses[-10:]) / max(1, len(result.losses[-10:])):.4f}\n"
            f"Final Capture Rate: {sum(result.captures[-10:]) / max(1, len(result.captures[-10:])):.1%}\n"
            f"Final Avg Steps: {sum(result.steps[-10:]) / max(1, len(result.steps[-10:])):.1f}\n"
            f"Best Checkpoint: {result.best_checkpoint or 'Not saved'}"
        )
        tk.Label(frame, text=info_text, fg=COLOR_SUCCESS, bg=COLOR_BACKGROUND, font=("Segoe UI", 12), justify=tk.LEFT).pack(pady=20)
        self._button(frame, "Run Visual Rollout", self.show_scenario_config)
        self._button(frame, "Main Menu", self.show_main_menu)

    def _show_train_error(self, error: Exception) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Training Failure", "Model training process failed.")
        tk.Label(frame, text=str(error), fg=COLOR_ERROR, bg=COLOR_BACKGROUND, font=("Segoe UI", 11)).pack(pady=20)
        self._button(frame, "Try Again", self.show_marl_training_config)
        self._button(frame, "Main Menu", self.show_main_menu)

    # Live Simulation Rendering

    def _max_steps_for_mode(self, rows: int, cols: int) -> int:
        if self.knowledge == KnowledgeMode.UNKNOWN:
            return UNKNOWN_MAX_STEPS_MULTIPLIER * rows * cols
        return 200

    def _create_environment(
        self,
        seed: int,
        grid_size: int,
        wall_density: float,
        barrel_density: float,
        mode: GameMode,
        knowledge: KnowledgeMode,
    ) -> PursuitEvasionEnv:
        spec = ScenarioSpec(
            grid_size=grid_size,
            wall_density=wall_density,
            barrel_density=barrel_density,
            seed=seed,
        )
        scenario = generate_scenario(spec, mode)
        environment = PursuitEvasionEnv(
            scenario,
            mode,
            max_steps=(
                UNKNOWN_MAX_STEPS_MULTIPLIER * scenario.rows * scenario.cols
                if knowledge == KnowledgeMode.UNKNOWN
                else 200
            ),
        )
        environment.reset()
        return environment

    def _new_environment(self) -> PursuitEvasionEnv:
        environment = self._create_environment(
            self.seed,
            self.custom_grid_size,
            self.custom_wall_density,
            self.custom_barrel_density,
            self.mode,
            self.knowledge,
        )
        self.perception = Perception() if self.knowledge == KnowledgeMode.UNKNOWN else None
        self.detection_step = 0 if self.knowledge == KnowledgeMode.KNOWN else None
        return environment

    def start_simulation(self, seed: int | None = None) -> None:
        self.seed = seed if seed is not None else SEED_GENERATOR.randrange(0, 2**32)
        self._simulation_generation += 1
        generation = self._simulation_generation
        self.running = False
        self._show_simulation_loading()
        Thread(
            target=self._load_simulation,
            args=(
                generation,
                self.seed,
                self.custom_grid_size,
                self.custom_wall_density,
                self.custom_barrel_density,
                self.mode,
                self.knowledge,
            ),
            daemon=True,
        ).start()

    def _show_simulation_loading(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Preparing Simulation", f"Generating seed {self.seed}...")
        tk.Label(
            frame,
            text="The simulation will appear when the scenario is ready.",
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BACKGROUND,
            font=("Segoe UI", 11),
        ).pack(pady=20)

    def _load_simulation(
        self,
        generation: int,
        seed: int,
        grid_size: int,
        wall_density: float,
        barrel_density: float,
        mode: GameMode,
        knowledge: KnowledgeMode,
    ) -> None:
        try:
            environment = self._create_environment(
                seed,
                grid_size,
                wall_density,
                barrel_density,
                mode,
                knowledge,
            )
        except (RuntimeError, ValueError) as error:
            self.root.after(
                0,
                lambda: self._finish_simulation_load(generation, None, error),
            )
            return
        self.root.after(
            0,
            lambda: self._finish_simulation_load(generation, environment, None),
        )

    def _finish_simulation_load(
        self,
        generation: int,
        environment: PursuitEvasionEnv | None,
        error: Exception | None,
    ) -> None:
        if generation != self._simulation_generation:
            return
        if error is not None:
            self._show_runtime_error("Could not create the scenario", error)
            return
        if environment is None:
            self._show_runtime_error(
                "Could not create the scenario",
                RuntimeError("Scenario generation returned no environment."),
            )
            return

        self.environment = environment
        self.perception = Perception() if self.knowledge == KnowledgeMode.UNKNOWN else None
        self.detection_step = 0 if self.knowledge == KnowledgeMode.KNOWN else None
        self.elapsed = 0.0
        self.started_at = perf_counter()
        self.paused = False
        self.running = True
        self._build_simulation_screen()
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _build_simulation_screen(self) -> None:
        self._clear()
        self.canvas = tk.Canvas(self.root, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(18, 8), pady=18)

        panel = tk.Frame(self.root, bg=COLOR_BACKGROUND, width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 18), pady=18)
        panel.pack_propagate(False)

        tk.Label(panel, text="SIMULATION", fg=COLOR_TEXT_MAIN, bg=COLOR_BACKGROUND, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(8, 10))

        self.run_status = tk.Label(panel, text="Starting...", fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, justify=tk.LEFT, anchor="w")
        self.run_status.pack(fill=tk.X, pady=(0, 10))

        self.checkpoint_badge = tk.Label(panel, text="", fg=COLOR_TEXT_MUTED, bg=COLOR_BACKGROUND, justify=tk.LEFT, anchor="w")
        self.checkpoint_badge.pack(fill=tk.X, pady=(0, 10))

        self.result_status = tk.Label(panel, text="", fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, justify=tk.LEFT, anchor="w")
        self.result_status.pack(fill=tk.X, pady=(0, 14))

        self.pause_button = ttk.Button(panel, text="Pause", command=self.toggle_pause, style="Action.TButton")
        self.pause_button.pack(fill=tk.X, pady=4)

        self.ground_truth_button = ttk.Button(
            panel,
            text="Show ground truth",
            command=self._toggle_ground_truth,
            style="Action.TButton",
        )
        self.ground_truth_button.pack(fill=tk.X, pady=4)

        ttk.Button(panel, text="New Seed", command=self.start_simulation, style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Restart", command=lambda: self.start_simulation(self.seed), style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Main Menu", command=self.show_main_menu, style="Action.TButton").pack(fill=tk.X, pady=4)

        legend = tk.Label(
            panel,
            text=(
                "Green: Pursuers\nRed: Visible evader\nGold: Known goal\n"
                "Brown: Known barrel\nBlue line: Known wall\n"
                "Dark: Unexplored\nGrey: Contaminated\nTinted green: Cleared"
            ),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BACKGROUND,
            justify=tk.LEFT,
            anchor="w",
        )
        legend.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

    def _toggle_ground_truth(self) -> None:
        self.show_ground_truth = not self.show_ground_truth
        if hasattr(self, "ground_truth_button"):
            self.ground_truth_button.config(
                text="Hide ground truth" if self.show_ground_truth else "Show ground truth"
            )
        self._draw()

    def _cell_box(self, position: Position) -> tuple[float, float, float, float]:
        row, col = position
        left = self._offset_x + col * self._cell_size
        top = self._offset_y + row * self._cell_size
        return left, top, left + self._cell_size, top + self._cell_size

    def _draw_circle(self, position: Position, inset: float, fill: str, outline: str, width: int = 2) -> None:
        left, top, right, bottom = self._cell_box(position)
        margin = self._cell_size * inset
        self.canvas.create_oval(left + margin, top + margin, right - margin, bottom - margin, fill=fill, outline=outline, width=width)

    def _draw_grid(self) -> None:
        if not self.environment:
            return
        if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None:
            belief = self.perception.pursuer_belief
            for r in range(self.environment.scenario.rows):
                for c in range(self.environment.scenario.cols):
                    position = (r, c)
                    if position in belief.contaminated_cells:
                        fill = "#cbd5e1"
                    elif position in belief.cleared_cells:
                        fill = "#244b3a"
                    elif position in belief.explored_cells:
                        fill = "#1e293b"
                    else:
                        fill = "#0b1220"
                    self.canvas.create_rectangle(*self._cell_box(position), fill=fill, outline="#334155")
            return
        for r in range(self.environment.scenario.rows):
            for c in range(self.environment.scenario.cols):
                self.canvas.create_rectangle(*self._cell_box((r, c)), fill="#1e293b", outline="#334155")

    def _draw_goals(self) -> None:
        if self.environment:
            goals = self.environment.scenario.goals
            if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None:
                goals = tuple(self.perception.pursuer_belief.known_goals)
            for g in goals:
                self.canvas.create_rectangle(*self._cell_box(g), fill="#d4a72c", outline="#f8d66d")

    def _draw_barrels(self) -> None:
        if self.environment:
            barrels = self.environment.scenario.barrels
            if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None:
                barrels = self.perception.pursuer_belief.known_barrels
            for b in barrels:
                self._draw_circle(b, inset=0.16, fill="#8b5e3c", outline="#d9a066")

    def _draw_walls(self) -> None:
        if not self.environment:
            return
        line_width = max(3, int(self._cell_size * 0.08))
        walls = self.environment.scenario.walls
        if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None:
            walls = self.perception.pursuer_belief.known_walls
        for pos, direction in walls:
            left, top, right, bottom = self._cell_box(pos)
            edges = {"up": (left, top, right, top), "down": (left, bottom, right, bottom), "left": (left, top, left, bottom), "right": (right, top, right, bottom)}
            self.canvas.create_line(*edges[direction], fill="#60a5fa", width=line_width)

    def _draw_pursuers(self) -> None:
        if self.environment:
            for p in self.environment.state.pursuers:
                self._draw_circle(p, inset=0.20, fill="#39d98a", outline="#b7f7d0")

    def _draw_evaders(self) -> None:
        if self.environment:
            if self.knowledge != KnowledgeMode.UNKNOWN or self.perception is None:
                for e in self.environment.state.evaders:
                    self._draw_circle(e, inset=0.18, fill="#f25f5c", outline="#ffd1d0")
                return
            if self.show_ground_truth:
                for e in self.environment.state.evaders:
                    self._draw_circle(e, inset=0.18, fill="#f25f5c", outline="#ffd1d0")
                return
            belief = self.perception.pursuer_belief
            for e in belief.visible_evaders:
                self._draw_circle(e, inset=0.18, fill="#f25f5c", outline="#ffd1d0")
            if belief.last_seen_evader is not None:
                last_seen = belief.last_seen_evader[0]
                if last_seen not in belief.visible_evaders:
                    left, top, right, bottom = self._cell_box(last_seen)
                    self.canvas.create_oval(
                        left + self._cell_size * 0.25,
                        top + self._cell_size * 0.25,
                        right - self._cell_size * 0.25,
                        bottom - self._cell_size * 0.25,
                        outline="#f25f5c",
                        width=3,
                        stipple="gray50",
                    )

    def _draw_ground_truth_overlay(self) -> None:
        if (
            not self.environment
            or self.knowledge != KnowledgeMode.UNKNOWN
            or not self.show_ground_truth
        ):
            return
        scenario = self.environment.scenario
        for barrel in scenario.barrels:
            self._draw_circle(barrel, inset=0.16, fill="#8b5e3c", outline="#f8d66d", width=2)
        for goal in scenario.goals:
            left, top, right, bottom = self._cell_box(goal)
            self.canvas.create_rectangle(
                left, top, right, bottom, outline="#f8d66d", width=2, stipple="gray50"
            )
        line_width = max(2, int(self._cell_size * 0.05))
        for position, direction in scenario.walls:
            left, top, right, bottom = self._cell_box(position)
            edges = {
                "up": (left, top, right, top),
                "down": (left, bottom, right, bottom),
                "left": (left, top, left, bottom),
                "right": (right, top, right, bottom),
            }
            self.canvas.create_line(*edges[direction], fill="#a78bfa", width=line_width, stipple="gray50")
        for evader in self.environment.state.evaders:
            self._draw_circle(evader, inset=0.18, fill="", outline="#f25f5c", width=3)

    def _draw(self) -> None:
        if not self.environment:
            return

        self.canvas.delete("all")
        scenario = self.environment.scenario
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        self._cell_size = min((canvas_w - 20) / scenario.cols, (canvas_h - 20) / scenario.rows)
        self._offset_x = (canvas_w - self._cell_size * scenario.cols) / 2
        self._offset_y = (canvas_h - self._cell_size * scenario.rows) / 2

        self._draw_grid()
        self._draw_ground_truth_overlay()
        self._draw_goals()
        self._draw_barrels()
        self._draw_walls()
        self._draw_pursuers()
        self._draw_evaders()

    def tick(self) -> None:
        if not self.running or not self.environment:
            return

        if not self.paused and not self.environment.done:
            try:
                if self.knowledge == KnowledgeMode.UNKNOWN:
                    pursuer_actions, evader_actions, is_trained, checkpoint_message = _autonomous_actions(
                        self.environment,
                        self.algorithm,
                        self.checkpoint_path,
                        self.knowledge,
                        self.perception,
                        self.viper_searcher,
                        self.switch_on_detection,
                    )
                else:
                    pursuer_actions, evader_actions, is_trained, checkpoint_message = _autonomous_actions(
                        self.environment,
                        self.algorithm,
                        self.checkpoint_path,
                        self.knowledge,
                        self.perception,
                    )
                self.is_checkpoint_active = is_trained
                self.checkpoint_message = checkpoint_message
                if self.knowledge == KnowledgeMode.UNKNOWN and hasattr(self, "viper_searcher"):
                    self.checkpoint_badge.config(text=self.viper_searcher.status_text)
                if (
                    self.knowledge == KnowledgeMode.UNKNOWN
                    and self.perception is not None
                    and self.perception.pursuer_belief.visible_evaders
                    and self.detection_step is None
                ):
                    self.detection_step = self.environment.state.step_count
                self.environment.step(pursuer_actions, evader_actions)
            except (RuntimeError, ValueError, IndexError) as error:
                self.running = False
                self._show_runtime_error("The scenario could not continue", error)
                return

            if self.environment.done:
                self.elapsed += perf_counter() - self.started_at
                self.started_at = 0.0
            else:
                self.elapsed = perf_counter() - self.started_at

        self._draw()
        self._update_status_panel()
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _show_runtime_error(self, title: str, error: Exception) -> None:
        self.running = False
        self._clear()
        frame = self._frame()
        self._title(frame, title, "The application is still running.")
        tk.Label(frame, text=str(error), fg=COLOR_ERROR, bg=COLOR_BACKGROUND, wraplength=650, justify=tk.LEFT, font=("Segoe UI", 11)).pack(pady=20)
        self._button(frame, "Try Again", self.show_scenario_config)
        self._button(frame, "Main Menu", self.show_main_menu)

    def _update_status_panel(self) -> None:
        if not self.environment:
            return
        scenario = self.environment.scenario
        terminal_info = self._terminal_info()

        result = self._result_text(terminal_info) if terminal_info else ("Paused" if self.paused else "Running autonomously")

        status_lines = [
            f"Mode: {MODE_LABELS[self.mode]}",
            f"Knowledge: {self.knowledge.value}",
            f"Algorithm: {self.algorithm}",
            f"Seed: {self.seed}",
            f"Grid {scenario.rows} x {scenario.cols}",
            f"Pursuers: {len(scenario.pursuers)}",
            f"Objects: {len(scenario.barrels)} barrels, {len(scenario.walls)} walls",
            f"Max Steps: {self.environment.max_steps}",
            (
                f"Explored: {self.perception.pursuer_belief.coverage():.1f}%"
                if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None
                else "Explored: 100.0%"
            ),
            (
                f"Cleared: {len(self.perception.pursuer_belief.cleared_cells) / max(1, scenario.rows * scenario.cols) * 100:.1f}%"
                if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None
                else "Cleared: 100.0%"
            ),
            (
                f"Evader Visible: {'Yes' if self.perception.pursuer_belief.visible_evaders else 'No'}"
                if self.knowledge == KnowledgeMode.UNKNOWN and self.perception is not None
                else "Evader Visible: Yes"
            ),
            (
                f"First Detection: {self.detection_step if self.detection_step is not None else 'Not detected'}"
                if self.knowledge == KnowledgeMode.UNKNOWN
                else "First Detection: 0"
            ),
            f"Time: {self.elapsed:.1f}s",
            f"Step: {self.environment.state.step_count}",
        ]
        self.run_status.config(text="\n".join(status_lines))

        if self.knowledge == KnowledgeMode.UNKNOWN:
            search_status = (
                self.viper_searcher.status_text
                if hasattr(self, "viper_searcher")
                else "Search: Fallback (0)"
            )
            self.checkpoint_badge.config(text=search_status, fg=COLOR_TEXT_LIGHT)
        elif self.algorithm in MARL_ALGORITHMS:
            if self.is_checkpoint_active:
                self.checkpoint_badge.config(text="Policy: Trained Weights", fg=COLOR_SUCCESS)
            elif self.checkpoint_message:
                self.checkpoint_badge.config(
                    text=f"Policy: Warning: {self.checkpoint_message}",
                    fg=COLOR_FAILURE,
                )
            else:
                self.checkpoint_badge.config(text="Policy: Warning: Fallback Policy", fg=COLOR_FAILURE)
        else:
            self.checkpoint_badge.config(text="Policy: Classical Planner", fg=COLOR_TEXT_LIGHT)

        result_color = COLOR_TEXT_LIGHT
        if terminal_info:
            result_color = COLOR_SUCCESS if terminal_info["reached_goal"] else COLOR_FAILURE

        self.result_status.config(text=result, fg=result_color)

    def _terminal_info(self) -> dict | None:
        if not self.environment or not self.environment.done:
            return None

        state, scenario = self.environment.state, self.environment.scenario
        captured = any(e in state.pursuers for e in state.evaders)
        reached_goal = any(e in scenario.goals for e in state.evaders)
        return {"captured": captured, "reached_goal": reached_goal}

    def _result_text(self, info: dict) -> str:
        if info["reached_goal"]:
            return "SUCCESS\nEvader reached the goal"
        if info["captured"]:
            return "FAILED\nEvader was captured"
        return "FAILED\nEvader did not reach the goal"

    def toggle_pause(self) -> None:
        if not self.environment or self.environment.done:
            return

        now = perf_counter()
        self.paused = not self.paused

        if self.paused:
            self.elapsed += now - self.started_at
            self.started_at = 0.0
            self.pause_button.config(text="Resume")
        else:
            self.started_at = now
            self.pause_button.config(text="Pause")

    def run(self) -> None:
        self.root.mainloop()


def run_gui(seed: int = 0) -> None:
    app = PursuitEvasionApp(seed)
    app.run()