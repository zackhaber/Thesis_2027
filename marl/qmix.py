from torch import Tensor, nn

from .base import Mixer


class QMIXMixer(Mixer):
    """Monotonic state-conditioned mixing network."""

    def __init__(self, agent_count: int, state_size: int):
        super().__init__(agent_count, state_size)
        self.weights = nn.Sequential(nn.Linear(state_size, agent_count), nn.Softplus())
        self.bias = nn.Linear(state_size, 1)

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        return (self.weights(state) * values).sum(dim=-1, keepdim=True) + self.bias(state)
