from env.level_generator import DifficultySpec, Scenario, generate_level

DEFAULT_SCENARIOS = [DifficultySpec(level=level, seed=100 + level) for level in (1, 2, 3)]


def make_scenarios(specs=DEFAULT_SCENARIOS) -> list[Scenario]:
    return [generate_level(spec) for spec in specs]
