from torch import Tensor

from .qmix import QMIXMixer


class QPLEXMixer(QMIXMixer):
    """QMIX backbone with advantage-based duplex dueling."""

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        weights = self.weights(state)
        mixed = super().forward(values, state)
        advantage = values - values.mean(dim=-1, keepdim=True)
        return mixed + (weights * advantage).sum(dim=-1, keepdim=True)
