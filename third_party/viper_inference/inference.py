"""Checkpoint loading and graph-pointer inference."""

from pathlib import Path

import torch

from .constants import EMBEDDING_DIM, NODE_INPUT_DIM
from .model import PolicyNet
from .observation import build_observation


class ViperPolicy:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        if "policy_model" not in payload:
            raise KeyError("ViPER checkpoint has no 'policy_model' state dictionary")
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(self.device)
        self.network.load_state_dict(payload["policy_model"])
        self.network.eval()

    def choose(self, position, explored, rows, cols, barrels, walls, other_positions=()):
        choices = self.choose_many(
            [position], explored, rows, cols, barrels, walls, other_positions)
        return choices[0]

    def choose_many(self, positions, explored, rows, cols, barrels, walls, other_positions=()):
        observations = [
            build_observation(position, explored, rows, cols, barrels, walls, other_positions)
            for position in positions
        ]
        tensors = [
            torch.cat([observation[0][index] for observation in observations], dim=0).to(self.device)
            for index in range(6)
        ]
        with torch.no_grad():
            logp = self.network(*tensors)
        choices = []
        for index, (_, candidates, diagnostics) in enumerate(observations):
            action_index = int(torch.argmax(logp[index], dim=0).item())
            choices.append((candidates[action_index], diagnostics))
        return choices