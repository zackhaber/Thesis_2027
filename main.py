"""
main.py

Central command-line interface and entry point for the Multi-Agent Pursuit-Evasion suite.
Dispatches subcommands for GUI visual rollouts, benchmark evaluations, MARL training, and notebooks.
"""

import argparse
import json
import subprocess
import sys
from notebooks import run_classical_suite
from ui import run_gui


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line interface subcommands and options."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent pursuit-evasion benchmarks and training pipelines."
    )
    subparsers = parser.add_subparsers(dest="command")

    # GUI Visual Rollout Demo
    demo = subparsers.add_parser("demo", help="Display the interactive GUI visual rollout.")
    demo.add_argument("--seed", type=int, default=0, help="Random seed for scenario generation.")

    # Classical Planners Benchmark Suite khghgkg
    benchmark = subparsers.add_parser("benchmark", help="Run the classical planner benchmark suite.")
    benchmark.add_argument("--grid-size", type=int, default=20)
    benchmark.add_argument("--wall-density", type=float, default=0.5)
    benchmark.add_argument("--barrel-density", type=float, default=0.1)
    benchmark.add_argument("--seeds", type=int, nargs="+", default=(0,))

    # Offline MARL Model Training
    train = subparsers.add_parser("train", help="Run MARL training pipeline and export model weights.")
    train.add_argument("--algorithm", choices=("IQL", "VDN", "QMIX", "QTRAN", "QPLEX"), default="QMIX")
    train.add_argument("--episodes", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--grid-size", type=int, default=20)
    train.add_argument("--pursuers", type=int, default=4)
    train.add_argument("--wall-density", type=float, default=0.75)
    train.add_argument("--barrel-density", type=float, default=0.0)
    train.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save .pt weights.")

    # Jupyter Notebook Launcher
    notebook = subparsers.add_parser("notebook", help="Open project notebooks in Jupyter.")
    notebook.add_argument("--path", default=".")
    return parser

def main() -> int:
    """Dispatches CLI commands to project modules."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo":
        run_gui(args.seed)
        return 0

    if args.command == "benchmark":
        results = run_classical_suite(
            grid_size=args.grid_size,
            wall_density=args.wall_density,
            barrel_density=args.barrel_density,
            seeds=tuple(args.seeds),
        )
        print(json.dumps(results, indent=2))
        return 0

    if args.command == "train":
        try:
            from marl import build_agent
        except ModuleNotFoundError as error:
            if error.name == "torch":
                parser.error("PyTorch is required for training. Install it with: pip install torch")
            raise

        # Initialize trainer with explicit checkpoint directory
        trainer = build_agent(
            args.algorithm,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            grid_size=args.grid_size,
            pursuers=args.pursuers,
            wall_density=args.wall_density,
            barrel_density=args.barrel_density,
        )
        result = trainer.train(args.episodes)

        summary = {
            "algorithm": result.algorithm,
            "seed": args.seed,
            "episodes": len(result.rewards),
            "final_reward": round(result.rewards[-1], 2) if result.rewards else 0.0,
            "final_loss": round(result.losses[-1], 4) if result.losses else 0.0,
            "checkpoint": result.checkpoint,
            "best_checkpoint": result.best_checkpoint,
            "final_evaluation_capture_rate": (
                result.evaluation_capture_rates[-1]
                if result.evaluation_capture_rates
                else None
            ),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "notebook":
        return subprocess.call([sys.executable, "-m", "jupyter", "notebook", args.path])

    # Default fallback: Launch the interactive Tkinter application
    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())