"""
ui.py

Graphical User Interface and evaluation controller for Pursuit-Evasion scenarios.
Integrates Visual Rollouts, Batch Benchmarks, and Offline MARL Model Training.
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

from env import PursuitEvasionEnv, ScenarioSpec, generate_scenario
from env.env import Action
from env.game_modes import GameMode
from marl.trainer import TorchMixer
from planners import cbs_plan, mstar_plan, voronoi_plan

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
    """Computes evasive step avoiding pursuers while advancing toward goals."""
    goal_mode = environment.mode in (GameMode.REACH_AVOID, GameMode.MAZE)

    if goal_mode:
        nearest_goal = _nearest_target(evader, environment.scenario.goals)
        safe_action = _first_step_to_target(environment, evader, nearest_goal, pursuer_positions)
        if safe_action is not None:
            return safe_action

    legal = _legal_actions(environment, evader)
    if not legal:
        return MOVEMENT_ACTIONS[0]

    def score_action(action: Action) -> tuple[int, int]:
        new_pos = _position_after_action(environment, evader, action)
        safety = min(_distance(new_pos, p) for p in pursuer_positions)
        goal_bonus = -_distance(new_pos, _nearest_target(evader, environment.scenario.goals)) if goal_mode else 0
        return (safety, goal_bonus)

    return max(legal, key=score_action)


def _planner_scenario(environment: PursuitEvasionEnv):
    """Creates scenario copy anchored to current agent positions."""
    return replace(
        environment.scenario,
        pursuers=tuple(environment.state.pursuers),
        evaders=tuple(environment.state.evaders),
    )


def _marl_pursuer_actions(
    environment: PursuitEvasionEnv,
    algorithm: str,
) -> tuple[list[Action], bool]:
    """Loads PyTorch checkpoint and returns pursuer actions plus checkpoint status."""
    checkpoint = Path("checkpoints") / f"{algorithm.lower()}_{environment.scenario.seed}_best.pt"
    if not checkpoint.exists():
        checkpoint = Path("checkpoints") / f"{algorithm.lower()}_{environment.scenario.seed}.pt"
    if not checkpoint.exists():
        return [_pursuer_action(environment, p) for p in environment.state.pursuers], False

    import torch
    from marl.base import AgentNetwork, state_tensor

    observation = environment.observe()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent_count = int(payload.get("agent_count", len(observation["pursuers"])))
    if agent_count != len(observation["pursuers"]):
        return [_pursuer_action(environment, p) for p in environment.state.pursuers], False
    policy = AgentNetwork(len(observation["global_state"]), agent_count)
    policy.load_state_dict(payload["agents"])
    policy.eval()

    with torch.no_grad():
        values = policy(
            state_tensor(observation).repeat(agent_count, 1),
            torch.arange(agent_count),
        )

    actions: list[Action] = []
    for index, mask in enumerate(observation["action_masks"]["pursuers"]):
        masked = values[index].clone()
        masked[~torch.tensor(mask, dtype=torch.bool)] = -torch.inf
        actions.append(ACTION_NAMES[int(masked.argmax())])
    return actions, True


def _planner_action(value: object, fallback: Action) -> Action:
    if isinstance(value, str) and value in ACTION_NAMES:
        return cast(Action, value)
    return fallback


def _autonomous_actions(
    environment: PursuitEvasionEnv,
    algorithm: str = "CBS",
) -> tuple[list[Action], list[Action], bool]:
    """Computes coordinated timestep moves for all agents."""
    is_trained = True
    pursuer_actions: list[Action]
    if algorithm in {"CBS", "M*", "Voronoi"}:
        planner = {"CBS": cbs_plan, "M*": mstar_plan, "Voronoi": voronoi_plan}[algorithm]
        result = planner(_planner_scenario(environment))
        pursuer_actions = [
            _planner_action(
                path[0] if path else None,
                _pursuer_action(environment, p),
            )
            for p, path in zip(environment.state.pursuers, result["paths"])
        ]
    else:
        pursuer_actions, is_trained = _marl_pursuer_actions(environment, algorithm)

    predicted_positions: set[Position] = set()
    for pursuer, action in zip(environment.state.pursuers, pursuer_actions):
        next_pos = _position_after_action(environment, pursuer, action, environment.scenario.pursuer_speed)
        predicted_positions.add(pursuer)
        predicted_positions.add(next_pos)

    evader_actions: list[Action] = [
        _evader_action(environment, evader, predicted_positions)
        for evader in environment.state.evaders
    ]

    return pursuer_actions, evader_actions, is_trained


# Application Interface
class PursuitEvasionApp:
    """Main application supporting visual rollouts, batch benchmarks, and MARL training."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.mode = GameMode.PURE_PURSUIT
        self.algorithm = "CBS"
        self.custom_grid_size = 20
        self.custom_wall_density = 0.5
        self.custom_barrel_density = 0.1

        self.environment: PursuitEvasionEnv | None = None
        self.running = False
        self.paused = False
        self.is_checkpoint_active = False

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
            add_label(7, "Number of Simulations")
            self.batch_runs = ttk.Entry(form, width=28, justify="center")
            self.batch_runs.insert(0, "10")
            self.batch_runs.grid(row=7, column=1, padx=10, pady=5)

        self.checkpoint_status_label = tk.Label(frame, text="", fg=COLOR_TEXT_MUTED, bg=COLOR_BACKGROUND, font=("Segoe UI", 10))
        self.checkpoint_status_label.pack(pady=4)

        self.config_error = tk.Label(frame, text="", fg=COLOR_ERROR, bg=COLOR_BACKGROUND, font=("Segoe UI", 10))
        self.config_error.pack(pady=4)
        return form

    def _update_config_algorithms(self) -> None:
        values = CLASSICAL_ALGORITHMS if self.config_family.get() == "Planner" else MARL_ALGORITHMS
        self.config_algorithm.configure(values=values)
        self.config_algorithm.set(values[0])
        self._check_checkpoint_status()

    def _check_checkpoint_status(self) -> None:
        if self.config_family.get() == "Planner":
            self.checkpoint_status_label.config(text="ℹ️ Classical Planners execute exact search paths dynamically.", fg=COLOR_TEXT_LIGHT)
            return

        alg = self.config_algorithm.get().lower()
        try:
            seed_val = int(self.config_seed_entry.get().strip())
        except ValueError:
            seed_val = self.seed

        ckpt_path = Path("checkpoints") / f"{alg}_{seed_val}.pt"
        if ckpt_path.exists():
            self.checkpoint_status_label.config(text=f"🟢 Checkpoint Found: {ckpt_path}", fg=COLOR_SUCCESS)
        else:
            self.checkpoint_status_label.config(text=f"⚠️ No Checkpoint Found ({ckpt_path}). Will use fallback/random policy.", fg=COLOR_FAILURE)

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
        self.algorithm = self.config_algorithm.get()
        self.seed = seed_val
        self.custom_grid_size = size
        self.custom_wall_density = walls
        self.custom_barrel_density = barrels
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
            capture_steps: list[int] = []
            planning_seconds: list[float] = []
            episode_seconds: list[float] = []

            for index in range(runs):
                episode_seed = (base_seed + index) % (2**32)
                scenario = generate_scenario(
                    ScenarioSpec(grid_size=self.custom_grid_size, wall_density=self.custom_wall_density, barrel_density=self.custom_barrel_density, seed=episode_seed),
                    self.mode,
                )
                environment = PursuitEvasionEnv(scenario, self.mode, seed=episode_seed)
                environment.reset()
                episode_started = perf_counter()
                info: dict = {"captured": False, "time": 0}

                while not environment.done:
                    planning_started = perf_counter()
                    pursuer_actions, evader_actions, _ = _autonomous_actions(environment, self.algorithm)
                    planning_seconds.append(perf_counter() - planning_started)
                    _, _, _, _, info = environment.step(pursuer_actions, evader_actions)

                episode_seconds.append(perf_counter() - episode_started)
                if bool(info.get("captured")):
                    captured += 1
                    capture_steps.append(int(info.get("time", 0)))

            results = (
                captured / runs * 100,
                sum(capture_steps) / len(capture_steps) if capture_steps else 0.0,
                sum(planning_seconds) / len(planning_seconds) * 1000,
                sum(episode_seconds) / len(episode_seconds) * 1000,
            )
            self.root.after(0, lambda: self._show_batch_results(results, runs))
        except Exception as error:
            self.root.after(0, lambda: self._show_batch_error(error))

    def _show_batch_results(self, results: tuple[float, float, float, float], runs: int) -> None:
        success_rate, average_steps, latency_ms, average_time_ms = results
        self._clear()
        frame = self._frame()
        self._title(frame, "Benchmark Results Summary", f"{self.algorithm} ({runs} Runs) | Mode: {MODE_LABELS[self.mode]}")
        result_text = (
            f"• Success Rate (%): {success_rate:.1f}%\n\n"
            f"• Path Efficiency (Avg Steps to Capture): {average_steps:.1f}\n\n"
            f"• Latency / Decision Time per Step: {latency_ms:.3f} ms\n\n"
            f"• Total Average Episode Execution Time: {average_time_ms:.3f} ms"
        )
        tk.Label(frame, text=result_text, fg=COLOR_TEXT_LIGHT, bg=COLOR_BACKGROUND, justify=tk.LEFT, font=("Segoe UI", 12)).pack(pady=20)
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
        self.train_seed.insert(0, "7")
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
            args=(alg, seed, episodes, grid_size, pursuers),
            daemon=True,
        ).start()

    def _run_training_worker(self, alg: str, seed: int, episodes: int, grid_size: int, pursuers: int) -> None:
        try:
            trainer = TorchMixer(
                algorithm=alg,
                seed=seed,
                checkpoint_dir="checkpoints",
                grid_size=grid_size,
                pursuers=pursuers,
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

    def _new_environment(self) -> PursuitEvasionEnv:
        spec = ScenarioSpec(grid_size=self.custom_grid_size, wall_density=self.custom_wall_density, barrel_density=self.custom_barrel_density, seed=self.seed)
        scenario = generate_scenario(spec, self.mode)
        environment = PursuitEvasionEnv(scenario, self.mode)
        environment.reset()
        return environment

    def start_simulation(self, seed: int | None = None) -> None:
        self.seed = seed if seed is not None else SEED_GENERATOR.randrange(0, 2**32)

        try:
            self.environment = self._new_environment()
        except (RuntimeError, ValueError) as error:
            self._show_runtime_error("Could not create the scenario", error)
            return

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

        ttk.Button(panel, text="New Seed", command=self.start_simulation, style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Restart", command=lambda: self.start_simulation(self.seed), style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Main Menu", command=self.show_main_menu, style="Action.TButton").pack(fill=tk.X, pady=4)

        legend = tk.Label(panel, text="Green: Pursuers\nRed: Evader\nGold: Goal\nBrown: Barrel\nBlue line: Wall", fg=COLOR_TEXT_MUTED, bg=COLOR_BACKGROUND, justify=tk.LEFT, anchor="w")
        legend.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

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
        for r in range(self.environment.scenario.rows):
            for c in range(self.environment.scenario.cols):
                self.canvas.create_rectangle(*self._cell_box((r, c)), fill="#1e293b", outline="#334155")

    def _draw_goals(self) -> None:
        if self.environment:
            for g in self.environment.scenario.goals:
                self.canvas.create_rectangle(*self._cell_box(g), fill="#d4a72c", outline="#f8d66d")

    def _draw_barrels(self) -> None:
        if self.environment:
            for b in self.environment.scenario.barrels:
                self._draw_circle(b, inset=0.16, fill="#8b5e3c", outline="#d9a066")

    def _draw_walls(self) -> None:
        if not self.environment:
            return
        line_width = max(3, int(self._cell_size * 0.08))
        for pos, direction in self.environment.scenario.walls:
            left, top, right, bottom = self._cell_box(pos)
            edges = {"up": (left, top, right, top), "down": (left, bottom, right, bottom), "left": (left, top, left, bottom), "right": (right, top, right, bottom)}
            self.canvas.create_line(*edges[direction], fill="#60a5fa", width=line_width)

    def _draw_pursuers(self) -> None:
        if self.environment:
            for p in self.environment.state.pursuers:
                self._draw_circle(p, inset=0.20, fill="#39d98a", outline="#b7f7d0")

    def _draw_evaders(self) -> None:
        if self.environment:
            for e in self.environment.state.evaders:
                self._draw_circle(e, inset=0.18, fill="#f25f5c", outline="#ffd1d0")

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
                pursuer_actions, evader_actions, is_trained = _autonomous_actions(self.environment, self.algorithm)
                self.is_checkpoint_active = is_trained
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
            f"Algorithm: {self.algorithm}",
            f"Seed: {self.seed}",
            f"Grid {scenario.rows} x {scenario.cols}",
            f"Pursuers: {len(scenario.pursuers)}",
            f"Objects: {len(scenario.barrels)} barrels, {len(scenario.walls)} walls",
            f"Time: {self.elapsed:.1f}s",
            f"Step: {self.environment.state.step_count}",
        ]
        self.run_status.config(text="\n".join(status_lines))

        if self.algorithm in MARL_ALGORITHMS:
            if self.is_checkpoint_active:
                self.checkpoint_badge.config(text="Policy: 🟢 Trained Weights", fg=COLOR_SUCCESS)
            else:
                self.checkpoint_badge.config(text="Policy: ⚠️ Fallback Policy", fg=COLOR_FAILURE)
        else:
            self.checkpoint_badge.config(text="Policy: 🔷 Classical Planner", fg=COLOR_TEXT_LIGHT)

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


def run_gui(seed: int = 7) -> None:
    app = PursuitEvasionApp(seed)
    app.run()