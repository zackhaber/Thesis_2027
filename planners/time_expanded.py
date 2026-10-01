from .cbs import cbs_plan


def time_expanded_plan(scenario):
    result = cbs_plan(scenario)
    result["algorithm"] = "TimeExpandedCBS"
    return result
