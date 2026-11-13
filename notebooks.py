"""Benchmark helpers shared by the command-line interface and notebooks."""

from typing import cast

from env import ScenarioSpec, generate_scenario
from env.env import Action
from env.game_modes import GameMode
from evaluation import evaluate_policy
from planners import cbs_plan, mstar_plan, voronoi_plan


def run_classical_suite(
    grid_size: int = 20,
    wall_density: float = 0.5,
    barrel_density: float = 0.1,
    seeds: tuple[int, ...] = (0,),
) -> list[dict]:
    """Evaluate each classical planner on the requested deterministic scenarios."""
    rows: list[dict] = []
    for seed in seeds:
        spec = ScenarioSpec(
            grid_size=grid_size,
            wall_density=wall_density,
            barrel_density=barrel_density,
            seed=seed,
        )
        scenario = generate_scenario(spec, GameMode.PURE_PURSUIT)
        for planner in (cbs_plan, mstar_plan, voronoi_plan):
            result = planner(scenario)
            plan = cast(list[list[Action]], result["paths"])

            def policy(
                observation: dict,
                plan: list[list[Action]] = plan,
            ) -> tuple[list[Action], list[Action]]:
                pursuer_actions = [
                    path.pop(0) if path else "stay" for path in plan
                ]
                evader_actions: list[Action] = ["stay"] * len(observation["evaders"])
                return pursuer_actions, evader_actions

            metrics = evaluate_policy(policy, [spec], mode=GameMode.PURE_PURSUIT)
            rows.append(
                {
                    "seed": seed,
                    "algorithm": result["algorithm"],
                    "planning_time": result["planning_time"],
                    **metrics,
                }
            )
    return rows
