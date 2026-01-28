import numpy as np
from collections import deque
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytiling import Tilemap


class DijkstraGrid:
    """
    Generates a distance gradient map using Dijkstra's algorithm (Flood Fill/BFS)
    starting from the target position. This map represents the walking distance
    from any reachable tile to the target, respecting physical obstacles.
    """

    def __init__(self, tilemap: "Tilemap", target_pos: tuple[int, int]):
        """
        Initializes the grid and computes the distance map immediately.

        Args:
            tilemap: The game's tilemap object.
            target_pos: Tuple (x, y) representing the target tile coordinates.
        """
        self.width, self.height = tilemap.grid_size

        # Initialize grid with Infinity to represent unreachable areas
        self.distance_grid = np.full(
            (self.width, self.height), np.inf, dtype=np.float32
        )

        self._compute_map(tilemap, target_pos)

    def _compute_map(self, tilemap: "Tilemap", target_pos: tuple[int, int]):
        """
        Performs Breadth-First Search (BFS) to populate the distance grid.
        """
        target_x, target_y = int(target_pos[0]), int(target_pos[1])

        # Validate target bounds
        if not (0 <= target_x < self.width and 0 <= target_y < self.height):
            logging.error(f"Dijkstra target {target_pos} is out of bounds.")
            return

        queue = deque([(target_x, target_y, 0)])
        self.distance_grid[target_x, target_y] = 0

        visited = set()
        visited.add((target_x, target_y))

        # 4-Directional movement (Up, Down, Left, Right)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        platforms_layer = tilemap.get_layer("platforms")

        while queue:
            curr_x, curr_y, dist = queue.popleft()

            new_dist = dist + 1

            for dx, dy in directions:
                nx, ny = curr_x + dx, curr_y + dy

                # Check boundaries
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                if (nx, ny) in visited:
                    continue

                # Check for walls/platforms as requested
                # If there is a tile, it is a wall (blocked)
                if platforms_layer.get_tile_at((nx, ny)) is not None:
                    continue

                # Valid path found
                self.distance_grid[nx, ny] = new_dist
                visited.add((nx, ny))
                queue.append((nx, ny, new_dist))

    def get_distance(self, x: float, y: float) -> float:
        """
        Returns the pre-calculated distance from the given coordinates to the target.
        Handles float coordinates by casting to int.
        Returns -1.0 if the position is out of bounds or unreachable (Infinity).
        """
        ix, iy = int(x), int(y)

        if 0 <= ix < self.width and 0 <= iy < self.height:
            dist = self.distance_grid[ix, iy]
            # Check if unreachable
            if dist == np.inf:
                return -1.0
            return float(dist)

        return -1.0
