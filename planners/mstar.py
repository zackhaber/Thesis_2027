from .cbs import cbs_plan


def mstar_plan(scenario) -> dict:
    # M* currently reuses the CBS baseline and changes only the reported label.
    result = cbs_plan(scenario)
    result["algorithm"] = "M*"
    return result
