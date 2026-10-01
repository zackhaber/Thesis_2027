from .cbs import cbs_plan


def time_expanded_plan(scenario):
    # This placeholder currently delegates to CBS instead of expanding time states.
    result = cbs_plan(scenario)
    result["algorithm"] = "TimeExpandedCBS"
    return result
