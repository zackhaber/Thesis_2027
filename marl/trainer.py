from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from env import PursuitEvasionEnv, ScenarioSpec, generate_scenario
from env.game_modes import GameMode

from .base import AgentNetwork, Mixer, TrainingResult, state_tensor
from .qmix import QMIXMixer
from .qplex import QPLEXMixer
from .qtran import QTRANMixer
from .vdn import VDNMixer

SUPPORTED_ALGORITHMS = {"IQL", "VDN", "QMIX", "QTRAN", "QPLEX"}
ACTION_NAMES = ("stay", "up", "down", "left", "right")
ProgressCallback = Callable[[int, int, float, float, bool, int, float, float], None]
TRAINING_MODE = GameMode.PURE_PURSUIT
DISTANCE_REWARD_SCALE = 0.05
TRAINING_WALL_DENSITY = 0.2
TRAINING_BARREL_DENSITY = 0.0
LEARNING_RATE = 0.00025
REPLAY_CAPACITY = 10000
REPLAY_BATCH_SIZE = 32
TARGET_UPDATE_INTERVAL = 5
EVALUATION_INTERVAL = 25
EVALUATION_EPISODES = 10


@dataclass
class Transition:
    state: Tensor
    actions: Tensor
    masks: Tensor
    next_state: Tensor
    next_masks: Tensor
    reward: float
    terminal: bool


def _build_mixer(algorithm: str, agent_count: int, state_size: int) -> Mixer:
    mixer_types: dict[str, type[Mixer]] = {
        "IQL": VDNMixer,
        "VDN": VDNMixer,
        "QMIX": QMIXMixer,
        "QTRAN": QTRANMixer,
        "QPLEX": QPLEXMixer,
    }
    return mixer_types[algorithm](agent_count, state_size)


def _masked_max(q_values: Tensor, masks: Tensor) -> Tensor:
    masked_values = q_values.masked_fill(~masks, -torch.inf)
    return masked_values.max(dim=1).values


def _nearest_evader_distance(observation: dict) -> int:
    pursuers = observation["pursuers"]
    evaders = observation["evaders"]
    return min(
        abs(pursuer[0] - evader[0]) + abs(pursuer[1] - evader[1])
        for pursuer in pursuers
        for evader in evaders
    )


class TorchMixer:
    """Train and export an identity-conditioned multi-agent value policy."""

    def __init__(
        self,
        algorithm: str,
        seed: int = 7,
        checkpoint_dir: str | None = None,
        grid_size: int = 20,
        pursuers: int = 4,
        wall_density: float = 0.5,
        barrel_density: float = 0.1,
    ):
        self.algorithm = algorithm.upper()
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
            raise ValueError(f"Supported algorithms: {supported}")
        if grid_size < 2 or pursuers < 1:
            raise ValueError("grid_size must be at least 2 and pursuers must be positive")
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.grid_size = grid_size
        self.pursuers = pursuers
        self.wall_density = wall_density
        self.barrel_density = barrel_density

    def train(
        self,
        episodes: int = 100,
        callback: ProgressCallback | None = None,
    ) -> TrainingResult:
        if episodes < 1:
            raise ValueError("episodes must be positive")

        torch.manual_seed(self.seed)
        scenario = generate_scenario(
            ScenarioSpec(
                grid_size=self.grid_size,
                pursuers=self.pursuers,
                seed=self.seed,
                wall_density=min(self.wall_density, TRAINING_WALL_DENSITY),
                barrel_density=min(self.barrel_density, TRAINING_BARREL_DENSITY),
            ),
            TRAINING_MODE,
        )
        environment = PursuitEvasionEnv(
            scenario,
            TRAINING_MODE,
            max_steps=max(100, self.grid_size * 2),
            seed=self.seed,
        )
        initial_observation = environment.reset()
        state_size = len(initial_observation["global_state"])
        agent_count = len(initial_observation["pursuers"])
        agents = AgentNetwork(state_size, agent_count)
        target_agents = AgentNetwork(state_size, agent_count)
        target_agents.load_state_dict(agents.state_dict())
        mixer = _build_mixer(self.algorithm, agent_count, state_size)
        target_mixer = _build_mixer(self.algorithm, agent_count, state_size)
        target_mixer.load_state_dict(mixer.state_dict())
        optimizer = torch.optim.Adam(
            list(agents.parameters()) + list(mixer.parameters()), lr=LEARNING_RATE
        )

        rewards: list[float] = []
        losses: list[float] = []
        captures: list[bool] = []
        steps: list[int] = []
        reward_moving_averages: list[float] = []
        loss_moving_averages: list[float] = []
        evaluation_capture_rates: list[float] = []
        evaluation_average_steps: list[float] = []
        replay_buffer: list[Transition] = []
        best_checkpoint: str | None = None
        best_capture_rate = -1.0
        agent_ids = torch.arange(agent_count)
        gamma = 0.99
        for episode in range(1, episodes + 1):
            episode_seed = self.seed + episode - 1
            scenario = generate_scenario(
                ScenarioSpec(
                    grid_size=self.grid_size,
                    pursuers=self.pursuers,
                    seed=episode_seed,
                    wall_density=min(self.wall_density, TRAINING_WALL_DENSITY),
                    barrel_density=min(self.barrel_density, TRAINING_BARREL_DENSITY),
                ),
                TRAINING_MODE,
            )
            environment = PursuitEvasionEnv(
                scenario,
                TRAINING_MODE,
                max_steps=max(100, self.grid_size * 2),
                seed=episode_seed,
            )
            observation = environment.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            updates = 0
            captured = False
            epsilon = max(0.05, 1.0 - 0.95 * (episode - 1) / max(episodes - 1, 1))

            while not environment.done:
                previous_distance = _nearest_evader_distance(observation)
                state = state_tensor(observation)
                state_per_agent = state.repeat(agent_count, 1)
                q_values = agents(state_per_agent, agent_ids)
                masks = torch.tensor(observation["action_masks"]["pursuers"], dtype=torch.bool)
                legal_q_values = q_values.masked_fill(~masks, -torch.inf)
                greedy_actions = legal_q_values.argmax(dim=1)
                random_actions = torch.tensor(
                    [
                        int(torch.where(mask)[0][torch.randint(int(mask.sum()), ())])
                        for mask in masks
                    ],
                    dtype=torch.long,
                )
                actions = (
                    random_actions
                    if torch.rand(()) < epsilon
                    else greedy_actions
                )
                next_observation, reward, terminated, truncated, info = environment.step(
                    [ACTION_NAMES[int(action)] for action in actions],
                    ["stay"] * len(observation["evaders"]),
                )
                next_distance = _nearest_evader_distance(next_observation)
                distance_progress = previous_distance - next_distance
                training_reward = (
                    reward["pursuers"]
                    + DISTANCE_REWARD_SCALE * distance_progress
                )
                if info["captured"]:
                    captured = True
                    training_reward = 1.0

                next_state = state_tensor(next_observation)
                next_masks = torch.tensor(
                    next_observation["action_masks"]["pursuers"], dtype=torch.bool
                )
                replay_buffer.append(
                    Transition(
                        state.squeeze(0).detach(),
                        actions.detach(),
                        masks.detach(),
                        next_state.squeeze(0).detach(),
                        next_masks.detach(),
                        training_reward,
                        terminated or truncated,
                    )
                )
                if len(replay_buffer) > REPLAY_CAPACITY:
                    replay_buffer.pop(0)

                if len(replay_buffer) >= REPLAY_BATCH_SIZE:
                    batch = [
                        replay_buffer[index]
                        for index in torch.randperm(len(replay_buffer))[
                            :REPLAY_BATCH_SIZE
                        ].tolist()
                    ]
                    batch_states = torch.stack([item.state for item in batch])
                    batch_actions = torch.stack([item.actions for item in batch])
                    batch_masks = torch.stack([item.masks for item in batch])
                    batch_next_states = torch.stack([item.next_state for item in batch])
                    batch_next_masks = torch.stack([item.next_masks for item in batch])
                    batch_rewards = torch.tensor(
                        [[item.reward] for item in batch], dtype=torch.float32
                    )
                    batch_terminal = torch.tensor(
                        [[item.terminal] for item in batch], dtype=torch.bool
                    )
                    batch_size = len(batch)
                    batch_agent_states = batch_states.repeat_interleave(agent_count, dim=0)
                    batch_agent_ids = agent_ids.repeat(batch_size)
                    batch_q_values = agents(batch_agent_states, batch_agent_ids).view(
                        batch_size, agent_count, -1
                    )
                    batch_chosen = batch_q_values.gather(
                        2, batch_actions.unsqueeze(-1)
                    ).squeeze(-1)
                    with torch.no_grad():
                        target_agent_values = target_agents(
                            batch_next_states.repeat_interleave(agent_count, dim=0),
                            batch_agent_ids,
                        ).view(batch_size, agent_count, -1)
                        target_best = _masked_max(
                            target_agent_values.view(-1, target_agent_values.shape[-1]),
                            batch_next_masks.view(-1, batch_next_masks.shape[-1]),
                        ).view(batch_size, agent_count)
                        target_joint = target_mixer(target_best, batch_next_states)
                        batch_target = batch_rewards + (
                            ~batch_terminal
                        ).float() * gamma * target_joint
                    joint_value = mixer(batch_chosen, batch_states)
                    loss = nn.functional.smooth_l1_loss(joint_value, batch_target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(agents.parameters()) + list(mixer.parameters()), 10.0
                    )
                    optimizer.step()
                    episode_loss += float(loss.item())
                    updates += 1
                episode_reward += training_reward
                observation = next_observation

            rewards.append(episode_reward)
            average_loss = episode_loss / max(updates, 1)
            losses.append(average_loss)
            episode_steps = environment.state.step_count
            captures.append(captured)
            steps.append(episode_steps)
            window_start = max(0, len(rewards) - 25)
            reward_average = sum(rewards[window_start:]) / len(rewards[window_start:])
            loss_average = sum(losses[window_start:]) / len(losses[window_start:])
            reward_moving_averages.append(reward_average)
            loss_moving_averages.append(loss_average)
            if episode % TARGET_UPDATE_INTERVAL == 0:
                target_agents.load_state_dict(agents.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
            evaluation_capture_rate = evaluation_capture_rates[-1] if evaluation_capture_rates else 0.0
            evaluation_steps = evaluation_average_steps[-1] if evaluation_average_steps else 0.0
            if episode % EVALUATION_INTERVAL == 0 or episode == episodes:
                evaluation_capture_rate, evaluation_steps = self._evaluate(
                    agents, state_size, agent_count, episode
                )
                evaluation_capture_rates.append(evaluation_capture_rate)
                evaluation_average_steps.append(evaluation_steps)
                if evaluation_capture_rate > best_capture_rate:
                    best_capture_rate = evaluation_capture_rate
                    best_checkpoint = self._save_checkpoint(
                        agents, mixer, state_size, agent_count, suffix="_best"
                    )
            if callback:
                callback(
                    episode,
                    episodes,
                    episode_reward,
                    average_loss,
                    captured,
                    episode_steps,
                    reward_average,
                    loss_average,
                )

        checkpoint = self._save_checkpoint(agents, mixer, state_size, agent_count)
        result = TrainingResult(
            self.algorithm,
            rewards,
            losses,
            checkpoint,
            captures,
            steps,
            reward_moving_averages,
            loss_moving_averages,
            best_checkpoint,
            evaluation_capture_rates,
            evaluation_average_steps,
        )
        self._write_results(result)
        return result

    def _evaluate(
        self,
        agents: AgentNetwork,
        state_size: int,
        agent_count: int,
        evaluation_number: int,
    ) -> tuple[float, float]:
        """Evaluate the current policy greedily on fresh scenarios."""
        captures = 0
        total_steps = 0
        agent_ids = torch.arange(agent_count)
        agents.eval()
        try:
            for index in range(EVALUATION_EPISODES):
                episode_seed = self.seed + 1_000_000 + evaluation_number * 100 + index
                scenario = generate_scenario(
                    ScenarioSpec(
                        grid_size=self.grid_size,
                        pursuers=self.pursuers,
                        seed=episode_seed,
                        wall_density=min(self.wall_density, TRAINING_WALL_DENSITY),
                        barrel_density=min(self.barrel_density, TRAINING_BARREL_DENSITY),
                    ),
                    TRAINING_MODE,
                )
                environment = PursuitEvasionEnv(
                    scenario,
                    TRAINING_MODE,
                    max_steps=max(100, self.grid_size * 2),
                    seed=episode_seed,
                )
                observation = environment.reset()
                captured = False
                while not environment.done:
                    state = state_tensor(observation)
                    with torch.no_grad():
                        values = agents(
                            state.repeat(agent_count, 1),
                            agent_ids,
                        )
                    actions = []
                    for values_for_agent, mask in zip(
                        values, observation["action_masks"]["pursuers"]
                    ):
                        legal_values = values_for_agent.masked_fill(
                            ~torch.tensor(mask, dtype=torch.bool), -torch.inf
                        )
                        actions.append(ACTION_NAMES[int(legal_values.argmax())])
                    observation, _, _, _, info = environment.step(
                        actions,
                        ["stay"] * len(observation["evaders"]),
                    )
                    captured = bool(info["captured"])
                captures += int(captured)
                total_steps += environment.state.step_count
        finally:
            agents.train()
        return captures / EVALUATION_EPISODES, total_steps / EVALUATION_EPISODES

    def _write_results(self, result: TrainingResult) -> None:
        """Overwrite the run report with the complete training history."""
        results_path = (
            self.checkpoint_dir / "results.txt"
            if self.checkpoint_dir is not None
            else Path("results.txt")
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        captures = result.captures or []
        steps = result.steps or []
        reward_averages = result.reward_moving_averages or []
        loss_averages = result.loss_moving_averages or []
        lines = [
            "MARL Training Results",
            "=====================",
            f"Algorithm: {result.algorithm}",
            f"Seed: {self.seed}",
            f"Episodes: {len(result.rewards)}",
            f"Grid size: {self.grid_size}",
            f"Pursuers: {self.pursuers}",
            f"Training mode: {TRAINING_MODE.value}",
            f"Maximum steps per episode: {max(100, self.grid_size * 2)}",
            f"Training wall density: {min(self.wall_density, TRAINING_WALL_DENSITY)}",
            f"Training barrel density: {min(self.barrel_density, TRAINING_BARREL_DENSITY)}",
            f"Distance reward scale: {DISTANCE_REWARD_SCALE}",
            f"Checkpoint: {result.checkpoint or 'Not saved'}",
            f"Best checkpoint: {result.best_checkpoint or 'Not saved'}",
            f"Learning rate: {LEARNING_RATE}",
            f"Replay capacity: {REPLAY_CAPACITY}",
            f"Replay batch size: {REPLAY_BATCH_SIZE}",
            f"Target update interval: {TARGET_UPDATE_INTERVAL} episodes",
            f"Evaluation interval: every {EVALUATION_INTERVAL} episodes",
            "",
            "Episode details",
            "---------------",
            "Episode | Seed | Reward | TD Loss | Captured | Steps | 25-Episode Avg Reward | 25-Episode Avg Loss",
        ]
        for index, (reward, loss) in enumerate(
            zip(result.rewards, result.losses), start=1
        ):
            capture = captures[index - 1] if index <= len(captures) else False
            episode_steps = steps[index - 1] if index <= len(steps) else 0
            reward_average = (
                reward_averages[index - 1]
                if index <= len(reward_averages)
                else reward
            )
            loss_average = (
                loss_averages[index - 1]
                if index <= len(loss_averages)
                else loss
            )
            lines.append(
                f"{index} | {self.seed + index - 1} | {reward:.6f} | {loss:.6f} | "
                f"{'Yes' if capture else 'No'} | {episode_steps} | "
                f"{reward_average:.6f} | {loss_average:.6f}"
            )

        recent_count = min(25, len(result.rewards))
        recent_rewards = result.rewards[-recent_count:]
        recent_losses = result.losses[-recent_count:]
        recent_captures = captures[-recent_count:]
        recent_steps = steps[-recent_count:]
        lines.extend(
            [
                "",
                "Summary",
                "-------",
                f"Final 25-episode average reward: {sum(recent_rewards) / max(1, len(recent_rewards)):.6f}",
                f"Final 25-episode average TD loss: {sum(recent_losses) / max(1, len(recent_losses)):.6f}",
                f"Final 25-episode capture rate: {sum(recent_captures) / max(1, len(recent_captures)):.2%}",
                f"Final 25-episode average steps: {sum(recent_steps) / max(1, len(recent_steps)):.2f}",
                f"Best checkpoint: {result.best_checkpoint or 'Not saved'}",
            ]
        )
        if result.evaluation_capture_rates:
            lines.extend(
                [
                    "",
                    "Greedy evaluation checkpoints",
                    "-----------------------------",
                    "Evaluation | Capture rate | Average steps",
                ]
            )
            for index, (capture_rate, average_steps) in enumerate(
                zip(
                    result.evaluation_capture_rates,
                    result.evaluation_average_steps or [],
                ),
                start=1,
            ):
                lines.append(
                    f"{index} | {capture_rate:.2%} | {average_steps:.2f}"
                )
        results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _save_checkpoint(
        self,
        agents: AgentNetwork,
        mixer: Mixer,
        state_size: int,
        agent_count: int,
        suffix: str = "",
    ) -> str | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.algorithm.lower()}_{self.seed}{suffix}.pt"
        torch.save(
            {
                "algorithm": self.algorithm,
                "agents": agents.state_dict(),
                "mixer": mixer.state_dict(),
                "state_size": state_size,
                "agent_count": agent_count,
                "grid_size": self.grid_size,
                "pursuers": self.pursuers,
                "training_mode": TRAINING_MODE.value,
                "distance_reward_scale": DISTANCE_REWARD_SCALE,
                "learning_rate": LEARNING_RATE,
                "replay_capacity": REPLAY_CAPACITY,
                "replay_batch_size": REPLAY_BATCH_SIZE,
                "target_update_interval": TARGET_UPDATE_INTERVAL,
                "best_checkpoint": suffix == "_best",
            },
            path,
        )
        return str(path)


def build_agent(
    name: str,
    seed: int = 7,
    checkpoint_dir: str | None = None,
    grid_size: int = 20,
    pursuers: int = 4,
    wall_density: float = 0.5,
    barrel_density: float = 0.1,
) -> TorchMixer:
    return TorchMixer(
        name,
        seed,
        checkpoint_dir,
        grid_size,
        pursuers,
        wall_density,
        barrel_density,
    )
