from typing import cast

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.game_modes import GameMode
from env.env import Action
from env.game_modes import GameMode
from planners import cbs_plan, mstar_plan, voronoi_plan
from evaluation import evaluate_policy


def make_environment(level: int = 1, seed: int = 7):
    return PursuitEvasionEnv(generate_level(DifficultySpec(level=level, seed=seed)))


def run_classical_suite(levels=(1, 2, 3)):
    rows = []
    for level in levels:
        # Use the level number as a stable seed so all planners share one scenario.
        scenario = generate_level(DifficultySpec(level=level, seed=level))
        for planner in (cbs_plan, mstar_plan, voronoi_plan):
            result = planner(scenario)
            plan = cast(list[list[Action]], result["paths"])

            def policy(observation: dict, plan: list[list[Action]] = plan) -> tuple[list[Action], list[Action]]:
                pursuer_actions: list[Action] = [path.pop(0) if path else "stay" for path in plan]
                evader_actions: list[Action] = ["stay"] * len(observation["evaders"])
                return pursuer_actions, evader_actions

            metrics = evaluate_policy(policy, [DifficultySpec(level=level, seed=level)], mode=GameMode.PURE_PURSUIT)
            rows.append({"level": level, "algorithm": result["algorithm"], "planning_time": result["planning_time"], **metrics})
    return rows
