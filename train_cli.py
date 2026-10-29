"""
Command-line interface for offline MARL network training.
"""

import argparse
from marl import build_agent

def main():
    parser = argparse.ArgumentParser(description="MARL Command-Line Training Script")
    parser.add_argument("--algorithm", type=str, default="QMIX", choices=["IQL", "VDN", "QMIX", "QTRAN", "QPLEX"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--pursuers", type=int, default=4)
    args = parser.parse_args()

    print(f"Starting {args.algorithm} training on seed {args.seed} for {args.episodes} episodes...")
    agent = build_agent(
        name=args.algorithm,
        seed=args.seed,
        checkpoint_dir="checkpoints",
        grid_size=args.grid_size,
        pursuers=args.pursuers,
    )
    def report(
        episode: int,
        total_episodes: int,
        reward: float,
        loss: float,
        captured: bool,
        steps: int,
        reward_average: float,
        loss_average: float,
    ) -> None:
        status = "Yes" if captured else "No"
        print(
            f"Episode {episode}/{total_episodes} | Reward: {reward:.2f} | "
            f"TD Loss: {loss:.4f} | Captured: {status} | Steps: {steps} | "
            f"25-Ep Avg Reward: {reward_average:.2f} | "
            f"25-Ep Avg Loss: {loss_average:.4f}"
        )

    result = agent.train(episodes=args.episodes, callback=report)
    recent_count = min(25, len(result.rewards))
    recent_slice = slice(-recent_count, None)
    recent_captures = result.captures[recent_slice] if result.captures else []
    recent_steps = result.steps[recent_slice] if result.steps else []
    print(
        f"Final 25-Episode Capture Rate: "
        f"{sum(recent_captures) / max(1, len(recent_captures)):.1%}"
    )
    print(
        f"Final 25-Episode Average Steps: "
        f"{sum(recent_steps) / max(1, len(recent_steps)):.1f}"
    )
    print(f"Training completed! Checkpoint saved to: {result.checkpoint}")

if __name__ == "__main__":
    main()