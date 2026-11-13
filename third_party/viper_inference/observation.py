"""Graph observation builder adapted from ViPER agent/node-manager code."""

from collections import deque

import numpy as np
import torch

from .constants import LOCAL_K_SIZE, LOCAL_MAP_CELLS, LOCAL_NODE_PADDING_SIZE, NODE_INPUT_DIM

_DIRECTIONS = ((-1, 0, "up", "down"), (1, 0, "down", "up"),
               (0, -1, "left", "right"), (0, 1, "right", "left"))


def connected(cell, other, walls, barrels):
    if other in barrels:
        return False
    delta = (other[0] - cell[0], other[1] - cell[1])
    for dr, dc, direction, opposite in _DIRECTIONS:
        if delta == (dr, dc):
            return (cell, direction) not in walls and (other, opposite) not in walls
    return False


def neighbours(cell, rows, cols, walls, barrels, allowed=None):
    result = []
    for dr, dc, _, _ in _DIRECTIONS:
        other = (cell[0] + dr, cell[1] + dc)
        if 0 <= other[0] < rows and 0 <= other[1] < cols and connected(cell, other, walls, barrels):
            if allowed is None or other in allowed:
                result.append(other)
    return result


def is_frontier(cell, explored, rows, cols, walls, barrels):
    return any(
        0 <= cell[0] + dr < rows and 0 <= cell[1] + dc < cols
        and (cell[0] + dr, cell[1] + dc) not in explored
        and connected(cell, (cell[0] + dr, cell[1] + dc), walls, barrels)
        for dr, dc, _, _ in _DIRECTIONS
    )


def build_observation(position, explored, rows, cols, barrels, walls, other_positions=()):
    known_free = {cell for cell in explored if cell not in barrels}
    known_free.add(position)
    radius = LOCAL_MAP_CELLS // 2
    local = {cell for cell in known_free
             if abs(cell[0] - position[0]) <= radius and abs(cell[1] - position[1]) <= radius}
    local.add(position)
    ordered = sorted(local)
    if len(ordered) > LOCAL_NODE_PADDING_SIZE:
        raise ValueError(f"ViPER local-node limit exceeded: {len(ordered)} > {LOCAL_NODE_PADDING_SIZE}")
    index = {cell: number for number, cell in enumerate(ordered)}
    candidates = list(dict.fromkeys([position] + neighbours(position, rows, cols, walls, barrels, local)))
    if len(candidates) > LOCAL_K_SIZE:
        raise ValueError(f"ViPER neighbour limit exceeded: {len(candidates)} > {LOCAL_K_SIZE}")

    features = []
    for cell in ordered:
        utility = float(is_frontier(cell, explored, rows, cols, walls, barrels))
        occupied = float(cell in other_positions and cell != position)
        features.append(((cell[1] - position[1]) / 10.0, (cell[0] - position[0]) / 10.0,
                         utility / 30.0, utility / 30.0, 0.0, 0.0, 0.0, occupied))
    padded = np.zeros((LOCAL_NODE_PADDING_SIZE, NODE_INPUT_DIM), dtype=np.float32)
    padded[:len(features)] = np.asarray(features, dtype=np.float32)
    node_padding = np.ones((1, 1, LOCAL_NODE_PADDING_SIZE), dtype=np.int16)
    node_padding[0, 0, :len(ordered)] = 0
    edge_mask = np.ones((LOCAL_NODE_PADDING_SIZE, LOCAL_NODE_PADDING_SIZE), dtype=np.int16)
    for cell, cell_index in index.items():
        edge_mask[cell_index, cell_index] = 0
        for neighbor in neighbours(cell, rows, cols, walls, barrels, local):
            edge_mask[cell_index, index[neighbor]] = 0
    edge_indices = np.asarray([index[cell] for cell in candidates], dtype=np.int64)
    padded_edges = np.zeros(LOCAL_K_SIZE, dtype=np.int64)
    padded_edges[:len(edge_indices)] = edge_indices
    edge_padding = np.ones((1, 1, LOCAL_K_SIZE), dtype=np.int16)
    edge_padding[0, 0, :len(edge_indices)] = 0
    tensors = [
        torch.from_numpy(padded).unsqueeze(0),
        torch.from_numpy(node_padding),
        torch.from_numpy(edge_mask).unsqueeze(0),
        torch.tensor([[[index[position]]]], dtype=torch.long),
        torch.from_numpy(padded_edges).reshape(1, LOCAL_K_SIZE, 1),
        torch.from_numpy(edge_padding),
    ]
    return tensors, candidates, {"local_nodes": len(ordered), "neighbours": len(candidates)}


def frontier_bfs(start, explored, rows, cols, barrels, walls):
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        cell, path = queue.popleft()
        if cell != start and is_frontier(cell, explored, rows, cols, walls, barrels):
            return path[0] if path else cell
        for neighbor in neighbours(cell, rows, cols, walls, barrels):
            if neighbor in explored and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return start