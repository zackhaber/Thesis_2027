import argparse
import json
import subprocess
import sys

from notebooks import run_classical_suite
from ui import run_gui

def build_parser() -> argparse.ArgumentParser:
    # Define the command-line entry points and their experiment-specific options.
    parser = argparse.ArgumentParser(description="Run the pursuit-evasion experiments.")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Display a generated environment.")
    demo.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    demo.add_argument("--seed", type=int, default=7)

    benchmark = subparsers.add_parser("benchmark", help="Run the classical planner benchmark.")
    benchmark.add_argument("--levels", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3))

    train = subparsers.add_parser("train", help="Run a lightweight MARL training baseline.")
    train.add_argument("--algorithm", choices=("IQL", "VDN", "QMIX", "QTRAN", "QPLEX"), default="VDN")
    train.add_argument("--episodes", type=int, default=100)
    train.add_argument("--seed", type=int, default=7)

    notebook = subparsers.add_parser("notebook", help="Open the project notebooks in Jupyter.")
    notebook.add_argument("--path", default="notebooks")
    return parser


def main() -> int:
    # Dispatch the selected command to the corresponding project driver.
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo":
        run_gui(args.level, args.seed)
        return 0
    if args.command == "benchmark":
        print(json.dumps(run_classical_suite(args.levels), indent=2))
        return 0
    if args.command == "train":
        try:
            from marl import build_agent
        except ModuleNotFoundError as error:
            if error.name == "torch":
                parser.error("PyTorch is required for training. Install it with: python -m pip install torch")
            raise
        result = build_agent(args.algorithm, args.seed).train(args.episodes)
        print(json.dumps({"algorithm": result.algorithm, "episodes": len(result.rewards), "final_reward": result.rewards[-1], "final_loss": result.losses[-1]}, indent=2))
        return 0
    if args.command == "notebook":
        return subprocess.call([sys.executable, "-m", "jupyter", "notebook", args.path])

    run_gui()
    return 0


if __name__ == "__main__":
    # Route command-line choices to the existing experiment drivers.
    raise SystemExit(main())
