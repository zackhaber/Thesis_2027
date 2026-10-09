from torch import Tensor

from .base import Mixer


class VDNMixer(Mixer):
    """Value decomposition network with additive joint value."""

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        return values.sum(dim=-1, keepdim=True)
