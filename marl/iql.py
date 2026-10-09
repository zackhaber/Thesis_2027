from torch import Tensor

from .base import Mixer


class IQLMixer(Mixer):
    """Independent Q-learning keeps each agent value separate."""

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        return values.mean(dim=-1, keepdim=True)
