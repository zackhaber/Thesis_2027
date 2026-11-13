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
    captures: list[bool] | None = None
    steps: list[int] | None = None
    reward_moving_averages: list[float] | None = None
    loss_moving_averages: list[float] | None = None
    best_checkpoint: str | None = None
    evaluation_capture_rates: list[float] | None = None
    evaluation_average_steps: list[float] | None = None


class AgentNetwork(nn.Module):
    def __init__(self, state_size: int, agent_count: int = 1):
        super().__init__()
        if agent_count < 1:
            raise ValueError("agent_count must be positive")
        self.state_size = state_size
        self.agent_count = agent_count
        self.layers = nn.Sequential(
            nn.Linear(state_size + agent_count, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def forward(self, state: Tensor, agent_indices: Tensor | None = None) -> Tensor:
        if state.ndim != 2:
            raise ValueError("state must have shape [batch, state_size]")
        if state.shape[1] != self.state_size:
            raise ValueError(f"expected state size {self.state_size}, got {state.shape[1]}")
        if agent_indices is None:
            if self.agent_count != 1:
                raise ValueError("agent_indices are required for multi-agent networks")
            agent_features = state.new_zeros((state.shape[0], 1))
        else:
            if agent_indices.ndim != 1 or agent_indices.shape[0] != state.shape[0]:
                raise ValueError("agent_indices must have one value per state row")
            if torch.any((agent_indices < 0) | (agent_indices >= self.agent_count)):
                raise ValueError("agent index is outside the configured agent count")
            agent_features = torch.nn.functional.one_hot(
                agent_indices.to(torch.long), num_classes=self.agent_count
            ).to(dtype=state.dtype)
        return self.layers(torch.cat((state, agent_features), dim=1))


class Mixer(nn.Module):
    def __init__(self, agent_count: int, state_size: int):
        super().__init__()
        self.agent_count = agent_count
        self.state_size = state_size

    def forward(self, values: Tensor, state: Tensor) -> Tensor:
        raise NotImplementedError

    def validate_inputs(self, values: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim != 2 or values.shape[1] != self.agent_count:
            raise ValueError("values must have shape [batch, agent_count]")
        if state.ndim != 2 or state.shape[1] != self.state_size:
            raise ValueError("state must have shape [batch, state_size]")
        return values, state


def state_tensor(observation: dict) -> Tensor:
    return torch.tensor(observation["global_state"], dtype=torch.float32).unsqueeze(0)


def belief_state_tensor(belief, env) -> Tensor:
    """Encode the team belief using the global-state layout plus explored."""
    state_vector: list[int] = []
    pursuers = set(belief.pursuer_positions)
    evaders = set(belief.visible_evaders)
    for row in range(belief.rows):
        for col in range(belief.cols):
            position = (row, col)
            state_vector.extend(
                (
                    int(position in pursuers),
                    int(position in evaders),
                    int(position in belief.known_barrels),
                    int(position in belief.known_goals),
                )
            )
            for direction in ("up", "down", "left", "right"):
                state_vector.append(int((position, direction) in belief.known_walls))
            state_vector.append(int(position in belief.explored_cells))
    return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)


def belief_action_masks(belief, env) -> tuple[tuple[bool, ...], ...]:
    """Build pursuer action masks using only known obstacles and walls."""
    moves = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}

    def move(position, action):
        if action == "stay":
            return position
        row, col = position
        for _ in range(env.scenario.pursuer_speed):
            next_position = (row + moves[action][0], col + moves[action][1])
            if not (0 <= next_position[0] < belief.rows and 0 <= next_position[1] < belief.cols):
                break
            if (
                ((position, action) in belief.known_walls)
                or ((next_position, opposites[action]) in belief.known_walls)
                or next_position in belief.known_barrels
            ):
                break
            position = next_position
        return position

    masks = []
    for position in belief.pursuer_positions:
        masks.append(
            tuple(
                True
                if action == "stay"
                else move(position, action) != position
                for action in ("stay", "up", "down", "left", "right")
            )
        )
    return tuple(masks)
