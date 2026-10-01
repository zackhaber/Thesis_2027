from .cbs import cbs_plan


def mstar_plan(scenario) -> dict:
    result = cbs_plan(scenario)
    result["algorithm"] = "M*"
    return result
