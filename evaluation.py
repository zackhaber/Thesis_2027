from dataclasses import asdict, dataclass
from statistics import mean, stdev
from time import perf_counter
from typing import Callable

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.env import Action
from env.game_modes import GameMode

Policy = Callable[[dict], tuple[list[Action], list[Action]]]


@dataclass(frozen=True)
class RolloutResult:
    captured: bool
    reached_goal: bool
    truncated: bool
    time_to_event: int
    path_efficiency: float
    compute_seconds: float


def rollout(policy: Policy, environment: PursuitEvasionEnv) -> RolloutResult:
    observation = environment.reset()
    initial_distance = min(
        abs(pursuer[0] - evader[0]) + abs(pursuer[1] - evader[1])
        for pursuer in observation["pursuers"]
        for evader in observation["evaders"]
    )
    distance_travelled = 0
    started = perf_counter()
    truncated = False
    info = {"captured": False, "reached_goal": False, "time": observation["step"]}
    while not environment.done:
        before = observation["pursuers"]
        pursuer_actions, evader_actions = policy(observation)
        observation, _, terminated, truncated, info = environment.step(pursuer_actions, evader_actions)
        distance_travelled += sum(
            abs(current[0] - previous[0]) + abs(current[1] - previous[1])
            for previous, current in zip(before, observation["pursuers"])
        )
        if terminated or truncated:
            break
    compute_seconds = perf_counter() - started
    captured = bool(info["captured"])
    path_efficiency = initial_distance / max(distance_travelled, initial_distance, 1)
    return RolloutResult(captured, bool(info["reached_goal"]), truncated, info["time"], path_efficiency, compute_seconds)


def confidence_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    margin = 1.96 * stdev(values) / (len(values) ** 0.5)
    return mean(values) - margin, mean(values) + margin


def evaluate_policy(policy: Policy, specs: list[DifficultySpec], mode: GameMode = GameMode.MAZE) -> dict:
    results = [rollout(policy, PursuitEvasionEnv(generate_level(spec, mode), mode)) for spec in specs]
    capture_rate = mean(result.captured for result in results) if results else 0.0
    capture_interval = confidence_interval([float(result.captured) for result in results])
    return {
        "episodes": len(results),
        "capture_rate": capture_rate,
        "capture_rate_ci95": capture_interval,
        "mean_time_to_event": mean(result.time_to_event for result in results) if results else 0.0,
        "mean_path_efficiency": mean(result.path_efficiency for result in results) if results else 0.0,
        "mean_compute_seconds": mean(result.compute_seconds for result in results) if results else 0.0,
        "results": [asdict(result) for result in results],
    }