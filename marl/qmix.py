import torch
from torch import Tensor, nn

from .base import Mixer


class QMIXMixer(Mixer):
    """Monotonic state-conditioned mixing network."""

    def __init__(self, agent_count: int, state_size: int, hidden_size: int = 64):
        super().__init__(agent_count, state_size)
        self.hidden_size = hidden_size
        self.hyper_w_1 = nn.Linear(state_size, agent_count * hidden_size)
        self.hyper_b_1 = nn.Linear(state_size, hidden_size)
        self.hyper_w_2 = nn.Linear(state_size, hidden_size)
        self.hyper_b_2 = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        values, state = self.validate_inputs(values, state)
        batch_size = values.shape[0]
        weights_1 = torch.abs(self.hyper_w_1(state)).view(batch_size, self.agent_count, self.hidden_size)
        bias_1 = self.hyper_b_1(state).view(batch_size, 1, self.hidden_size)
        hidden = torch.bmm(values.unsqueeze(1), weights_1) + bias_1
        hidden = torch.nn.functional.elu(hidden)
        weights_2 = torch.abs(self.hyper_w_2(state)).view(batch_size, self.hidden_size, 1)
        bias_2 = self.hyper_b_2(state).view(batch_size, 1, 1)
        return (torch.bmm(hidden, weights_2) + bias_2).squeeze(1)
