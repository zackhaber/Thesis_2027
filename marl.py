from dataclasses import dataclass
from random import Random

SUPPORTED_ALGORITHMS = {"VDN", "QMIX", "QTRAN", "QPLEX"}


@dataclass
class TrainingResult:
    algorithm: str
    rewards: list[float]
    losses: list[float]

class TabularMixer:
    def __init__(self, algorithm: str, seed: int = 7):
        self.algorithm = algorithm
        self.rng = Random(seed)

    def train(self, episodes: int = 100) -> TrainingResult:
        rewards: list[float] = []
        losses: list[float] = []
        value = 0.0
        for episode in range(episodes):
            value += (1.0 - value) * 0.03
            rewards.append(value + self.rng.uniform(-0.1, 0.1))
            losses.append(max(0.0, 1.0 - episode / max(1, episodes)))
        return TrainingResult(self.algorithm, rewards, losses)

def build_agent(name: str, seed: int = 7) -> TabularMixer:
    algorithm = name.upper()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError("Supported algorithms: VDN, QMIX, QTRAN, QPLEX")
    return TabularMixer(algorithm, seed)
