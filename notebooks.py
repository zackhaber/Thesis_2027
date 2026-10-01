from env import DifficultySpec, PursuitEvasionEnv, generate_level
from planners import cbs_plan, mstar_plan, voronoi_plan


def make_environment(level: int = 1, seed: int = 7):
    return PursuitEvasionEnv(generate_level(DifficultySpec(level=level, seed=seed)))


def run_classical_suite(levels=(1, 2, 3)):
    rows = []
    for level in levels:
        scenario = generate_level(DifficultySpec(level=level, seed=level))
        for planner in (cbs_plan, mstar_plan, voronoi_plan):
            result = planner(scenario)
            rows.append({"level": level, "algorithm": result["algorithm"], "planning_time": result["planning_time"], "success": result["success"]})
    return rows
