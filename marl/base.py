from dataclasses import dataclass

import torch
from torch import Tensor, nn

ACTION_COUNT = 5


@dataclass
class TrainingResult:
    algorithm: str
    rewards: list[float]
    losses: list[float]
    checkpoint: str | None = None


class AgentNetwork(nn.Module):
    def __init__(self, state_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, ACTION_COUNT))

    def forward(self, state: Tensor) -> Tensor:
        return self.layers(state)


class Mixer(nn.Module):
    def __init__(self, agent_count: int, state_size: int):
        super().__init__()
        self.agent_count = agent_count
        self.state_size = state_size

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        raise NotImplementedError


def state_tensor(observation: dict) -> Tensor:
    return torch.tensor(observation["global_state"], dtype=torch.float32).unsqueeze(0)
