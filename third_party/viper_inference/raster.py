"""Native-scale rasterization for project scenarios."""

from dataclasses import dataclass

import numpy as np

from .constants import FREE, NODE_RESOLUTION_PIXELS, OBSTACLE, PIXELS_PER_CELL


@dataclass(frozen=True)
class RasterMap:
    image: np.ndarray
    node_centres: dict[tuple[int, int], tuple[int, int]]
    wall_px: int


def rasterize(scenario, wall_px=3):
    """Render one project cell as one ViPER node spacing (10 pixels)."""
    height, width = scenario.rows * PIXELS_PER_CELL, scenario.cols * PIXELS_PER_CELL
    image = np.full((height, width), FREE, dtype=np.uint8)
    for row, col in scenario.barrels:
        image[row * PIXELS_PER_CELL:(row + 1) * PIXELS_PER_CELL,
              col * PIXELS_PER_CELL:(col + 1) * PIXELS_PER_CELL] = OBSTACLE
    half = max(1, wall_px // 2)
    for (row, col), direction in scenario.walls:
        if direction == "up":
            y, x = row * PIXELS_PER_CELL, col * PIXELS_PER_CELL
            image[max(0, y - half):y + half + 1, x:(col + 1) * PIXELS_PER_CELL] = OBSTACLE
        elif direction == "down":
            y, x = (row + 1) * PIXELS_PER_CELL, col * PIXELS_PER_CELL
            image[max(0, y - half):y + half + 1, x:(col + 1) * PIXELS_PER_CELL] = OBSTACLE
        elif direction == "left":
            x, y = col * PIXELS_PER_CELL, row * PIXELS_PER_CELL
            image[y:(row + 1) * PIXELS_PER_CELL, max(0, x - half):x + half + 1] = OBSTACLE
        elif direction == "right":
            x, y = (col + 1) * PIXELS_PER_CELL, row * PIXELS_PER_CELL
            image[y:(row + 1) * PIXELS_PER_CELL, max(0, x - half):x + half + 1] = OBSTACLE
    nodes = {
        (row, col): (col * PIXELS_PER_CELL + PIXELS_PER_CELL // 2,
                     row * PIXELS_PER_CELL + PIXELS_PER_CELL // 2)
        for row in range(scenario.rows) for col in range(scenario.cols)
        if (row, col) not in scenario.barrels
    }
    return RasterMap(image, nodes, wall_px)


def verify_scale():
    return {"cell_size_m": 0.4, "node_resolution_m": 4.0,
            "pixels_per_cell": PIXELS_PER_CELL,
            "node_resolution_pixels": NODE_RESOLUTION_PIXELS}