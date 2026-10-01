import tkinter as tk
from tkinter import ttk
from collections import deque
from random import SystemRandom
from time import perf_counter

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.env import Action
from env.game_modes import GameMode

MOVEMENT_ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")
MODE_LABELS = {
    GameMode.PURE_PURSUIT: "Pure Pursuit",
    GameMode.REACH_AVOID: "Reach and Avoid",
    GameMode.MAZE: "Maze",
}
CLASSICAL_ALGORITHMS = ("CBS", "M*", "Voronoi")
MARL_ALGORITHMS = ("VDN", "QMIX", "QTRAN", "QPLEX")
SIMULATION_INTERVAL_MS = 600
SEED_GENERATOR = SystemRandom()
DIFFICULTIES = {
    1: {"title": "      Level 1", "description": "5 x 5 grid | 2 pursuers | no objects"},
    2: {"title": "      Level 2", "description": "10 x 10 grid | 3 pursuers | walls and barrels"},
    3: {"title": "      Level 3", "description": "20 x 20 grid | 4 pursuers | many walls and barrels"},
}


def render_ascii(environment: PursuitEvasionEnv) -> str:
    # Draw layers in order so agents remain visible over board objects.
    state = environment.observe()
    grid = [["." for _ in range(environment.scenario.cols)] for _ in range(environment.scenario.rows)]
    for row, col in environment.scenario.barrels:
        grid[row][col] = "B"
    for row, col in environment.scenario.goals:
        grid[row][col] = "G"
    for row, col in state["evaders"]:
        grid[row][col] = "E"
    for row, col in state["pursuers"]:
        grid[row][col] = "P"
    return "\n".join(" ".join(row) for row in grid)


def _position_after_action(environment: PursuitEvasionEnv, position: tuple[int, int], action: Action, speed: int = 1) -> tuple[int, int]:
    return environment._move(position, action, speed)


def _distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _nearest_target(position: tuple[int, int], targets: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return min(targets, key=lambda target: _distance(position, target))


def _first_step_to_target(
    environment: PursuitEvasionEnv,
    start: tuple[int, int],
    target: tuple[int, int],
    blocked_positions: set[tuple[int, int]] | None = None,
) -> Action | None:
    # Search for a legal route and return only its first action for this timestep.
    queue: deque[tuple[tuple[int, int], Action | None]] = deque([(start, None)])
    visited = {start}
    blocked = blocked_positions or set()
    while queue:
        position, first_action = queue.popleft()
        if position == target:
            return first_action
        for action in MOVEMENT_ACTIONS:
            candidate = _position_after_action(environment, position, action, 1)
            if candidate == position or candidate in visited or candidate in blocked:
                continue
            visited.add(candidate)
            queue.append((candidate, first_action or action))
    return None


def _fallback_movement_action(environment: PursuitEvasionEnv, position: tuple[int, int], targets: tuple[tuple[int, int], ...] = ()) -> Action:
    # Choose a legal move when path search fails, favoring nearby targets if supplied.
    legal_actions: list[Action] = [
        action
        for action in MOVEMENT_ACTIONS
        if _position_after_action(environment, position, action) != position
    ]
    if not legal_actions:
        return MOVEMENT_ACTIONS[0]
    if not targets:
        return legal_actions[0]
    return min(
        legal_actions,
        key=lambda action: min(
            _distance(_position_after_action(environment, position, action), target)
            for target in targets
        ),
    )


def _pursuer_action(environment: PursuitEvasionEnv, pursuer: tuple[int, int]) -> Action:
    target = _nearest_target(pursuer, tuple(environment.state.evaders))
    return _first_step_to_target(environment, pursuer, target) or _fallback_movement_action(environment, pursuer, (target,))


def _evader_action(
    environment: PursuitEvasionEnv,
    evader: tuple[int, int],
    pursuer_positions: set[tuple[int, int]],
) -> Action:
    # Goal modes prioritize a safe goal route; otherwise maximize distance from danger.
    if environment.mode in (GameMode.REACH_AVOID, GameMode.MAZE):
        target = _nearest_target(evader, environment.scenario.goals)
        safe_path = _first_step_to_target(environment, evader, target, pursuer_positions)
        if safe_path is not None:
            return safe_path

    legal_actions: list[Action] = [
        action
        for action in MOVEMENT_ACTIONS
        if _position_after_action(environment, evader, action) != evader
    ]
    if not legal_actions:
        return MOVEMENT_ACTIONS[0]
    return max(
        legal_actions,
        key=lambda action: (
            min(
                _distance(_position_after_action(environment, evader, action), pursuer)
                for pursuer in pursuer_positions
            ),
            -_distance(_position_after_action(environment, evader, action), _nearest_target(evader, environment.scenario.goals))
            if environment.mode in (GameMode.REACH_AVOID, GameMode.MAZE)
            else 0,
        ),
    )


def _autonomous_actions(environment: PursuitEvasionEnv) -> tuple[list[Action], list[Action]]:
    # Evaders avoid both current and next-step positions predicted for pursuers.
    pursuer_actions: list[Action] = [_pursuer_action(environment, pursuer) for pursuer in environment.state.pursuers]
    predicted_pursuer_positions = {
        position
        for pursuer, action in zip(environment.state.pursuers, pursuer_actions)
        for position in (pursuer, _position_after_action(environment, pursuer, action, environment.scenario.pursuer_speed))
    }
    evader_actions: list[Action] = [
        _evader_action(environment, evader, predicted_pursuer_positions)
        for evader in environment.state.evaders
    ]
    return pursuer_actions, evader_actions


class PursuitEvasionApp:
    def __init__(self, initial_level: int = 1, seed: int = 7):
        # Store menu selections and simulation timing before creating the first screen.
        self.seed = seed
        self.level = initial_level
        self.mode = GameMode.PURE_PURSUIT
        self.algorithm = "CBS"
        self.selected_seed: int | None = None
        self.environment: PursuitEvasionEnv | None = None
        self.running = False
        self.paused = False
        self.started_at = 0.0
        self.elapsed = 0.0
        self.root = tk.Tk()
        self._configure_styles()
        self.root.title("Multi-Agent Strategies for \nPursuit-Evasion Games")
        self.root.configure(bg="#111827")
        self.root.minsize(900, 650)
        self.show_main_menu()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Action.TButton", anchor="center", justify="center", font=("Segoe UI", 11, "bold"), padding=(18, 11), foreground="#f8fafc", background="#2563eb", borderwidth=0)
        style.map("Action.TButton", background=[("active", "#3b82f6"), ("pressed", "#1d4ed8")], foreground=[("disabled", "#94a3b8")])

    def _clear(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def _frame(self) -> tk.Frame:
        frame = tk.Frame(self.root, bg="#111827")
        frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        return frame

    def _title(self, parent: tk.Misc, text: str, subtitle: str) -> None:
        tk.Label(parent, text=text, fg="#f8fafc", bg="#111827", font=("Segoe UI", 25, "bold")).pack(pady=(25, 8))
        tk.Label(parent, text=subtitle, fg="#94a3b8", bg="#111827", font=("Segoe UI", 11)).pack(pady=(0, 30))

    def _button(self, parent: tk.Misc, text: str, command, width: int = 28) -> None:
        ttk.Button(parent, text=text, command=command, width=width, style="Action.TButton").pack(pady=7)

    def show_main_menu(self) -> None:
        self.running = False
        self._clear()
        frame = self._frame()
        self._title(frame, "Multi-Agent Strategies for \nPursuit-Evasion Games", "By Zack Haber, 2027")
        self._button(frame, "Play", self.show_mode_menu)
        self._button(frame, "Quit", self.root.destroy)

    def show_mode_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose Game Mode", "Select the objective for the evader")
        self._button(frame, "Pure Pursuit\n(Survive until captured)", lambda: self.choose_mode(GameMode.PURE_PURSUIT))
        self._button(frame, "Reach and Avoid\n(Reach a goal before capture)", lambda: self.choose_mode(GameMode.REACH_AVOID))
        self._button(frame, "Maze\n(Navigate walls to goal or capture)", lambda: self.choose_mode(GameMode.MAZE))
        self._button(frame, "Back", self.show_main_menu)

    def choose_mode(self, mode: GameMode) -> None:
        self.mode = mode
        self.show_algorithm_menu()

    def show_algorithm_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose Algorithm", f"Mode: {MODE_LABELS[self.mode]}")
        columns = tk.Frame(frame, bg="#111827")
        columns.pack(fill=tk.X, expand=True, pady=(0, 18))
        classical_column = tk.Frame(columns, bg="#111827")
        classical_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))
        marl_column = tk.Frame(columns, bg="#111827")
        marl_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))
        tk.Label(classical_column, text="Classical Planners", fg="#93c5fd", bg="#111827", font=("Segoe UI", 11, "bold")).pack(pady=(0, 4))
        for algorithm in CLASSICAL_ALGORITHMS:
            self._button(classical_column, algorithm, lambda selected=algorithm: self.choose_algorithm(selected), 18)
        tk.Label(marl_column, text="MARL Algorithms", fg="#86efac", bg="#111827", font=("Segoe UI", 11, "bold")).pack(pady=(0, 4))
        for algorithm in MARL_ALGORITHMS:
            self._button(marl_column, algorithm, lambda selected=algorithm: self.choose_algorithm(selected), 18)
        self._button(frame, "Back", self.show_mode_menu)

    def choose_algorithm(self, algorithm: str) -> None:
        self.algorithm = algorithm
        self.show_seed_menu()

    def show_seed_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose Seed", f"Mode: {MODE_LABELS[self.mode]} | Algorithm: {self.algorithm}")
        self._button(frame, "Random Seed", self.use_random_seed)

        seed_entry = tk.Frame(frame, bg="#111827")
        seed_entry.pack(pady=(18, 4))
        tk.Label(seed_entry, text="Specific Seed", fg="#cbd5e1", bg="#111827", font=("Segoe UI", 11)).pack(pady=(0, 6))
        self.seed_input = ttk.Entry(seed_entry, width=28, justify="center")
        self.seed_input.pack(pady=(0, 6))
        self.seed_input.bind("<Return>", lambda _event: self.use_specific_seed())
        self.seed_error = tk.Label(seed_entry, text="", fg="#fca5a5", bg="#111827", font=("Segoe UI", 10))
        self.seed_error.pack()
        self._button(seed_entry, "Search", self.use_specific_seed, 18)
        self._button(frame, "Back", self.show_algorithm_menu)
        self.seed_input.focus_set()

    def use_random_seed(self) -> None:
        self.selected_seed = None
        self.show_difficulty_menu()

    def use_specific_seed(self) -> None:
        try:
            self.selected_seed = int(self.seed_input.get().strip())
        except ValueError:
            self.seed_error.config(text="Enter a valid integer seed")
            self.seed_input.focus_set()
            return
        self.show_difficulty_menu()

    def show_difficulty_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose Difficulty", f"Mode: {MODE_LABELS[self.mode]} | Algorithm: {self.algorithm}")
        for level, details in DIFFICULTIES.items():
            self._button(frame, f"{details['title']}\n{details['description']}", lambda selected=level: self.start_simulation(selected, self.selected_seed), 45)
        self._button(frame, "Back", self.show_seed_menu)

    def _new_environment(self) -> PursuitEvasionEnv:
        # Generate and reset a fresh environment from the selected level and seed.
        scenario = generate_level(DifficultySpec(level=self.level, seed=self.seed), self.mode)
        environment = PursuitEvasionEnv(scenario, self.mode)
        environment.reset()
        return environment

    def start_simulation(self, level: int, seed: int | None = None) -> None:
        # Use a cryptographically generated seed when the user did not supply one.
        self.level = level
        self.seed = seed if seed is not None else SEED_GENERATOR.randrange(0, 2**32)
        self.environment = self._new_environment()
        self.elapsed = 0.0
        self.started_at = perf_counter()
        self.paused = False
        self.running = True
        # Start the recurring Tkinter callback that advances the simulation.
        self._build_simulation_screen()
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _build_simulation_screen(self) -> None:
        # Build the board canvas and the status/control panel used during a run.
        self._clear()
        self.canvas = tk.Canvas(self.root, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(18, 8), pady=18)
        panel = tk.Frame(self.root, bg="#111827", width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 18), pady=18)
        panel.pack_propagate(False)
        tk.Label(panel, text="SIMULATION", fg="#f8fafc", bg="#111827", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(8, 15))
        self.run_status = tk.Label(panel, text="Starting...", fg="#cbd5e1", bg="#111827", justify=tk.LEFT, anchor="w")
        self.run_status.pack(fill=tk.X, pady=(0, 18))
        self.result_status = tk.Label(panel, text="", fg="#cbd5e1", bg="#111827", justify=tk.LEFT, anchor="w")
        self.result_status.pack(fill=tk.X, pady=(0, 18))
        self.pause_button = ttk.Button(panel, text="Pause", command=self.toggle_pause, style="Action.TButton")
        self.pause_button.pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="New Seed", command=lambda: self.start_simulation(self.level), style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Restart", command=lambda: self.start_simulation(self.level, self.seed), style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Main Menu", command=self.show_main_menu, style="Action.TButton").pack(fill=tk.X, pady=4)
        tk.Label(panel, text="Green: Pursuers\nRed: Evader\nGold: Goal\nBrown: Barrel\nBlue line: Wall", fg="#94a3b8", bg="#111827", justify=tk.LEFT, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, pady=8)

    def _draw(self) -> None:
        # Render the grid, goals, barrels, walls, pursuers, and evaders in layers.
        if self.environment is None:
            return
        self.canvas.delete("all")
        scenario = self.environment.scenario
        cell_size = min((self.canvas.winfo_width() - 20) / scenario.cols, (self.canvas.winfo_height() - 20) / scenario.rows)
        offset_x = (self.canvas.winfo_width() - cell_size * scenario.cols) / 2
        offset_y = (self.canvas.winfo_height() - cell_size * scenario.rows) / 2

        def box(position):
            row, col = position
            left = offset_x + col * cell_size
            top = offset_y + row * cell_size
            return left, top, left + cell_size, top + cell_size

        for row in range(scenario.rows):
            for col in range(scenario.cols):
                self.canvas.create_rectangle(*box((row, col)), fill="#1e293b", outline="#334155")
        for goal in scenario.goals:
            self.canvas.create_rectangle(*box(goal), fill="#d4a72c", outline="#f8d66d")
        for barrel in scenario.barrels:
            left, top, right, bottom = box(barrel)
            self.canvas.create_oval(left + cell_size * .16, top + cell_size * .16, right - cell_size * .16, bottom - cell_size * .16, fill="#8b5e3c", outline="#d9a066", width=2)
        for position, direction in scenario.walls:
            left, top, right, bottom = box(position)
            points = {"up": (left, top, right, top), "down": (left, bottom, right, bottom), "left": (left, top, left, bottom), "right": (right, top, right, bottom)}
            self.canvas.create_line(*points[direction], fill="#60a5fa", width=max(3, int(cell_size * .08)))
        for pursuer in self.environment.state.pursuers:
            left, top, right, bottom = box(pursuer)
            self.canvas.create_oval(left + cell_size * .2, top + cell_size * .2, right - cell_size * .2, bottom - cell_size * .2, fill="#39d98a", outline="#b7f7d0", width=2)
        for evader in self.environment.state.evaders:
            left, top, right, bottom = box(evader)
            self.canvas.create_oval(left + cell_size * .18, top + cell_size * .18, right - cell_size * .18, bottom - cell_size * .18, fill="#f25f5c", outline="#ffd1d0", width=2)

    def tick(self) -> None:
        # Advance, redraw, update status, and schedule the next GUI timer tick.
        if not self.running or self.environment is None:
            return
        if not self.paused and not self.environment.done:
            self.environment.step(*_autonomous_actions(self.environment))
            if self.environment.done:
                self.elapsed += perf_counter() - self.started_at
                self.started_at = 0.0
            else:
                self.elapsed = perf_counter() - self.started_at
        self._draw()
        scenario = self.environment.scenario
        terminal_info = self._terminal_info()
        result = self._result_text(terminal_info) if terminal_info else ("Paused" if self.paused else "Running autonomously")
        self.run_status.config(text=f"Mode: {MODE_LABELS[self.mode]}\nAlgorithm: {self.algorithm}\nSeed: {self.seed}\nLevel {self.level} | Grid {scenario.rows} x {scenario.cols}\nPursuers: {len(scenario.pursuers)}\nObjects: {len(scenario.barrels)} barrels, {len(scenario.walls)} walls\nTime: {self.elapsed:.1f}s\nStep: {self.environment.state.step_count}")
        result_color = "#22c55e" if terminal_info and terminal_info["reached_goal"] else "#f87171" if terminal_info else "#cbd5e1"
        self.result_status.config(text=result, fg=result_color)
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _terminal_info(self) -> dict | None:
        # Derive final outcome flags from the environment's settled board state.
        if self.environment is None or not self.environment.done:
            return None
        return {
            "captured": any(evader in self.environment.state.pursuers for evader in self.environment.state.evaders),
            "reached_goal": any(evader in self.environment.scenario.goals for evader in self.environment.state.evaders),
        }

    def _result_text(self, info: dict) -> str:
        # Convert terminal flags into the short message shown in the result label.
        if info["reached_goal"]:
            return "SUCCESS\nEvader reached the goal"
        if info["captured"]:
            return "FAILED\nEvader was captured"
        return "FAILED\nEvader did not reach the goal"

    def toggle_pause(self) -> None:
        # Pause timing without losing elapsed time, then resume from the current clock.
        if self.environment is None or self.environment.done:
            return
        now = perf_counter()
        self.paused = not self.paused
        if self.paused:
            self.elapsed += now - self.started_at
            self.started_at = 0.0
        else:
            self.started_at = now
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def run(self) -> None:
        self.root.mainloop()


def run_gui(level: int = 1, seed: int = 7) -> None:
    PursuitEvasionApp(level, seed).run()
