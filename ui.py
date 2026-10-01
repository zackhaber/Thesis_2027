import tkinter as tk
from tkinter import ttk
from collections import deque
from random import SystemRandom
from time import perf_counter

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.env import Action
from env.game_modes import GameMode

ACTION_ORDER: tuple[Action, ...] = ("stay", "up", "down", "left", "right")
MODE_LABELS = {
    GameMode.PURE_PURSUIT: "Pure pursuit",
    GameMode.REACH_AVOID: "Reach and avoid",
    GameMode.MAZE: "Maze",
}
CLASSICAL_ALGORITHMS = ("CBS", "M*", "Voronoi")
MARL_ALGORITHMS = ("VDN", "QMIX", "QTRAN", "QPLEX")
SIMULATION_INTERVAL_MS = 600
SEED_GENERATOR = SystemRandom()
DIFFICULTIES = {
    1: {"title": "Level 1", "description": "5 x 5 grid | 1 pursuer | no objects"},
    2: {"title": "Level 2", "description": "10 x 10 grid | 2 pursuers | walls and barrels"},
    3: {"title": "Level 3", "description": "20 x 20 grid | 3 pursuers | many walls and barrels"},
}


def render_ascii(environment: PursuitEvasionEnv) -> str:
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


def _first_step_to_target(environment: PursuitEvasionEnv, start: tuple[int, int], target: tuple[int, int]) -> Action | None:
    queue: deque[tuple[tuple[int, int], Action | None]] = deque([(start, None)])
    visited = {start}
    closest_distance = _distance(start, target)
    closest_first_action: Action | None = None
    while queue:
        position, first_action = queue.popleft()
        if position == target:
            return first_action if first_action is not None else "stay"
        if first_action is not None and closest_first_action is None:
            closest_first_action = first_action
        distance = _distance(position, target)
        if distance < closest_distance and first_action is not None:
            closest_distance = distance
            closest_first_action = first_action
        for action in ACTION_ORDER[1:]:
            candidate = _position_after_action(environment, position, action, 1)
            if candidate == position or candidate in visited:
                continue
            visited.add(candidate)
            queue.append((candidate, first_action or action))
    return closest_first_action


def _pursuer_action(environment: PursuitEvasionEnv, pursuer: tuple[int, int]) -> Action:
    target = _nearest_target(pursuer, tuple(environment.state.evaders))
    return _first_step_to_target(environment, pursuer, target) or "stay"


def _evader_action(environment: PursuitEvasionEnv, evader: tuple[int, int]) -> Action:
    if environment.mode in (GameMode.REACH_AVOID, GameMode.MAZE):
        target = _nearest_target(evader, environment.scenario.goals)
        return _first_step_to_target(environment, evader, target) or "stay"

    legal_actions: list[Action] = [
        action
        for action in ACTION_ORDER[1:]
        if _position_after_action(environment, evader, action) != evader
    ]
    candidates: list[Action] = legal_actions or ["stay"]
    return max(
        candidates,
        key=lambda action: min(
            _distance(_position_after_action(environment, evader, action), pursuer)
            for pursuer in environment.state.pursuers
        ),
    )


def _autonomous_actions(environment: PursuitEvasionEnv) -> tuple[list[Action], list[Action]]:
    pursuer_actions: list[Action] = [_pursuer_action(environment, pursuer) for pursuer in environment.state.pursuers]
    evader_actions: list[Action] = [_evader_action(environment, evader) for evader in environment.state.evaders]
    return pursuer_actions, evader_actions


class PursuitEvasionApp:
    def __init__(self, initial_level: int = 1, seed: int = 7):
        self.seed = seed
        self.level = initial_level
        self.mode = GameMode.PURE_PURSUIT
        self.algorithm = "CBS"
        self.environment: PursuitEvasionEnv | None = None
        self.running = False
        self.paused = False
        self.started_at = 0.0
        self.elapsed = 0.0
        self.root = tk.Tk()
        self._configure_styles()
        self.root.title("Multi-Agent Strategies for Pursuit-Evasion Games")
        self.root.configure(bg="#111827")
        self.root.minsize(900, 650)
        self.show_main_menu()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Action.TButton", anchor="center", font=("Segoe UI", 11, "bold"), padding=(18, 11), foreground="#f8fafc", background="#2563eb", borderwidth=0)
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
        self._title(frame, "Multi-Agent Strategies for Pursuit-Evasion Games", "By Zack Haber, 2027")
        self._button(frame, "Play", self.show_mode_menu)
        self._button(frame, "Quit", self.root.destroy)

    def show_mode_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose game mode", "Select the objective for the evader")
        self._button(frame, "Pure pursuit\n(Survive until captured)", lambda: self.choose_mode(GameMode.PURE_PURSUIT))
        self._button(frame, "Reach and avoid\n(Reach a goal before capture)", lambda: self.choose_mode(GameMode.REACH_AVOID))
        self._button(frame, "Maze\n(Navigate walls to goal or capture)", lambda: self.choose_mode(GameMode.MAZE))
        self._button(frame, "Back", self.show_main_menu)

    def choose_mode(self, mode: GameMode) -> None:
        self.mode = mode
        self.show_algorithm_menu()

    def show_algorithm_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose algorithm", f"Mode: {MODE_LABELS[self.mode]}")
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
        self.show_difficulty_menu()

    def show_difficulty_menu(self) -> None:
        self._clear()
        frame = self._frame()
        self._title(frame, "Choose difficulty", f"Mode: {MODE_LABELS[self.mode]} | Algorithm: {self.algorithm}")
        for level, details in DIFFICULTIES.items():
            self._button(frame, f"{details['title']}\n{details['description']}", lambda selected=level: self.start_simulation(selected), 45)
        self._button(frame, "Back", self.show_algorithm_menu)

    def _new_environment(self) -> PursuitEvasionEnv:
        scenario = generate_level(DifficultySpec(level=self.level, seed=self.seed), self.mode)
        environment = PursuitEvasionEnv(scenario, self.mode)
        environment.reset()
        return environment

    def start_simulation(self, level: int, seed: int | None = None) -> None:
        self.level = level
        self.seed = seed if seed is not None else SEED_GENERATOR.randrange(0, 2**32)
        self.environment = self._new_environment()
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
        panel = tk.Frame(self.root, bg="#111827", width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 18), pady=18)
        panel.pack_propagate(False)
        tk.Label(panel, text="SIMULATION", fg="#f8fafc", bg="#111827", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(8, 15))
        self.run_status = tk.Label(panel, text="Starting...", fg="#cbd5e1", bg="#111827", justify=tk.LEFT, anchor="w")
        self.run_status.pack(fill=tk.X, pady=(0, 18))
        self.pause_button = ttk.Button(panel, text="Pause", command=self.toggle_pause, style="Action.TButton")
        self.pause_button.pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="New Seed", command=lambda: self.start_simulation(self.level), style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Restart", command=lambda: self.start_simulation(self.level, self.seed), style="Action.TButton").pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Main Menu", command=self.show_main_menu, style="Action.TButton").pack(fill=tk.X, pady=4)
        tk.Label(panel, text="Green: pursuers\nRed: evader\nGold: goal\nBrown: barrel\nBlue line: wall", fg="#94a3b8", bg="#111827", justify=tk.LEFT, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, pady=8)

    def _draw(self) -> None:
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
        self.run_status.config(text=f"Mode: {MODE_LABELS[self.mode]}\nAlgorithm: {self.algorithm}\nSeed: {self.seed}\nLevel {self.level} | Grid {scenario.rows} x {scenario.cols}\nPursuers: {len(scenario.pursuers)}\nObjects: {len(scenario.barrels)} barrels, {len(scenario.walls)} walls\nTime: {self.elapsed:.1f}s\nStep: {self.environment.state.step_count}\n{result}")
        self.root.after(SIMULATION_INTERVAL_MS, self.tick)

    def _terminal_info(self) -> dict | None:
        if self.environment is None or not self.environment.done:
            return None
        return {
            "captured": any(evader in self.environment.state.pursuers for evader in self.environment.state.evaders),
            "reached_goal": any(evader in self.environment.scenario.goals for evader in self.environment.state.evaders),
        }

    def _result_text(self, info: dict) -> str:
        if info["reached_goal"]:
            return "SUCCESS\nEvader reached the goal"
        if info["captured"]:
            return "FAILED\nEvader was captured"
        return "FAILED\nEvader did not reach the goal"

    def toggle_pause(self) -> None:
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
