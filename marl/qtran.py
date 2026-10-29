from torch import Tensor

from .qmix import QMIXMixer


class QTRANMixer(QMIXMixer):
    """QMIX backbone with the QTRAN joint-value correction term."""

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        mixed = super().forward(values, state)
        correction = values.mean(dim=1, keepdim=True)
        return mixed + correction
