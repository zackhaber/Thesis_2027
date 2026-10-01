from pathlib import Path
import csv


def capture_metrics(info: dict, planning_time: float = 0.0) -> dict:
    return {"capture": int(info.get("captured", False)), "time_to_capture": info.get("time"), "planning_time": planning_time}


def save_metrics(rows: list[dict], path: str = "results/metrics.csv") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def scenario_summary(scenario) -> dict:
    return {"rows": scenario.rows, "cols": scenario.cols, "pursuers": len(scenario.pursuers), "evaders": len(scenario.evaders), "obstacles": len(scenario.obstacles), "goals": len(scenario.goals)}
