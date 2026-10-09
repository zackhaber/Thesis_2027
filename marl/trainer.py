# This file 

from pathlib import Path
from typing import Type

import torch
from torch import nn

from env import DifficultySpec, PursuitEvasionEnv, generate_level
from env.game_modes import GameMode

from .base import AgentNetwork, Mixer, TrainingResult, state_tensor
from .iql import IQLMixer
from .qmix import QMIXMixer
from .qplex import QPLEXMixer
from .qtran import QTRANMixer
from .vdn import VDNMixer

SUPPORTED_ALGORITHMS = {"IQL", "VDN", "QMIX", "QTRAN", "QPLEX"}
MIXERS: dict[str, Type[Mixer]] = {
    "IQL": IQLMixer,
    "VDN": VDNMixer,
    "QMIX": QMIXMixer,
    "QTRAN": QTRANMixer,
    "QPLEX": QPLEXMixer,
}


class TorchMixer:
    def __init__(self, algorithm: str, seed: int = 7, checkpoint_dir: str | None = None):
        self.algorithm = algorithm.upper()
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError("Supported algorithms: IQL, VDN, QMIX, QTRAN, QPLEX")
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def train(self, episodes: int = 100) -> TrainingResult:
        torch.manual_seed(self.seed)
        environment = PursuitEvasionEnv(
            generate_level(DifficultySpec(level=1, seed=self.seed), GameMode.MAZE),
            GameMode.MAZE,
            max_steps=40,
            seed=self.seed,
        )
        state_size = len(environment.reset()["global_state"])
        agent_count = len(environment.state.pursuers)
        agents = AgentNetwork(state_size)
        mixer = MIXERS[self.algorithm](agent_count, state_size)
        optimizer = torch.optim.Adam(list(agents.parameters()) + list(mixer.parameters()), lr=0.001)
        rewards: list[float] = []
        losses: list[float] = []
        action_names = ("stay", "up", "down", "left", "right")
        for _ in range(episodes):
            observation = environment.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            updates = 0
            while not environment.done:
                state = state_tensor(observation)
                q_values = agents(state).repeat(agent_count, 1)
                masks = torch.tensor(observation["action_masks"]["pursuers"], dtype=torch.bool)
                q_values = q_values.masked_fill(~masks, -torch.inf)
                actions = q_values.argmax(dim=-1).tolist()
                next_observation, reward, terminated, truncated, _ = environment.step(
                    [action_names[action] for action in actions], ["stay"] * len(observation["evaders"])
                )
                chosen = q_values.gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
                value = mixer(chosen.unsqueeze(0), state)
                target = torch.tensor([[reward["pursuers"]]], dtype=torch.float32)
                loss = (value - target).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
                updates += 1
                episode_reward += reward["pursuers"]
                observation = next_observation
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
            losses.append(episode_loss / max(updates, 1))
        checkpoint = None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / f"{self.algorithm.lower()}_{self.seed}.pt"
            torch.save({"algorithm": self.algorithm, "agents": agents.state_dict(), "mixer": mixer.state_dict()}, path)
            checkpoint = str(path)
        return TrainingResult(self.algorithm, rewards, losses, checkpoint)


def build_agent(name: str, seed: int = 7, checkpoint_dir: str | None = None) -> TorchMixer:
    return TorchMixer(name, seed, checkpoint_dir)
