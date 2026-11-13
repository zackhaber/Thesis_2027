"""Inference sensor adapted from third_party/ViPER/utils/sensor.py."""

from dataclasses import dataclass
from math import hypot


@dataclass
class RasterSensor:
    rows: int
    cols: int
    barrels: frozenset[tuple[int, int]]
    walls: frozenset[tuple[tuple[int, int], str]]
    radius_cells: int = 5

    def _wall(self, cell, other):
        direction_by_delta = {(-1, 0): "up", (1, 0): "down",
                              (0, -1): "left", (0, 1): "right"}
        delta = (other[0] - cell[0], other[1] - cell[1])
        if abs(delta[0]) == 1 and abs(delta[1]) == 1:
            return (self._wall(cell, (other[0], cell[1]))
                    or self._wall(cell, (cell[0], other[1])))
        if delta not in direction_by_delta:
            return True
        direction = direction_by_delta[delta]
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}[direction]
        return (cell, direction) in self.walls or (other, opposite) in self.walls

    def line_clear(self, start, end):
        """Check crossed cell edges, including diagonal rays."""
        x0, y0 = start[1] + 0.5, start[0] + 0.5
        x1, y1 = end[1] + 0.5, end[0] + 0.5
        samples = max(1, int(hypot(x1 - x0, y1 - y0) * 32))
        previous = start
        for index in range(1, samples + 1):
            ratio = index / samples
            cell = (int(y0 + (y1 - y0) * ratio), int(x0 + (x1 - x0) * ratio))
            if cell == previous:
                continue
            if not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols):
                return False
            if cell in self.barrels or self._wall(previous, cell):
                return False
            previous = cell
        return True

    def visible_cells(self, origin):
        visible = set()
        for row in range(max(0, origin[0] - self.radius_cells),
                         min(self.rows, origin[0] + self.radius_cells + 1)):
            for col in range(max(0, origin[1] - self.radius_cells),
                             min(self.cols, origin[1] + self.radius_cells + 1)):
                cell = (row, col)
                if abs(row - origin[0]) + abs(col - origin[1]) <= self.radius_cells + 1:
                    if self.line_clear(origin, cell):
                        visible.add(cell)
        return visible

    def observe(self, origins, explored):
        for origin in origins:
            explored.update(self.visible_cells(origin))
        return explored