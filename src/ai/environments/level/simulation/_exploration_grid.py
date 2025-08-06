import numpy as np
from typing import Tuple


class ExplorationGrid:
    """
    A grid that tracks visited locations, wrapping a NumPy array.

    This class provides a simple interface for marking and querying visited
    cells on a 2D grid, which is useful for exploration-based policies.
    Using composition by wrapping a NumPy array is generally safer and more
    flexible than subclassing np.ndarray directly.
    """

    def __init__(self, width: int, height: int):
        """
        Initializes a new VisitedGrid.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
        """
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("Grid dimensions must be positive integers.")
        self._grid = np.zeros((width, height), dtype=np.uint8)

    @property
    def grid(self) -> np.ndarray:
        """Returns the underlying NumPy grid."""
        return self._grid

    @property
    def shape(self) -> tuple:
        """Returns the shape of the grid."""
        return self._grid.shape

    def step_on(self, position: Tuple[int, int], radius: int = 0) -> bool:
        """
        Marks a tile and its surrounding area within a radius as "stepped on" (value 1).

        This is an internal method to update the grid's state. The check for
        whether the tile was newly stepped on is performed only on the central
        tile of the operation, not the entire radius.

        Args:
            position (Tuple[int, int]): The (x, y) coordinates of the tile to step on.
            radius (int): The radius around the position to mark as stepped on.
                          A radius of 0 affects only the single tile at `position`.

        Returns:
            bool: True if the central tile at `position` was not previously
                  stepped on (i.e., its value was 0), False otherwise.
        """
        x, y = int(position[0]), int(position[1])
        width, height = self.shape

        if not (0 <= x < width and 0 <= y < height):
            # Stepping outside the grid cannot be a "new" step.
            return False

        # Check the status of the central tile *before* modification.
        is_new_step = self._grid[x, y] == 0

        # Define the bounding box for efficiency
        min_x = max(0, x - radius)
        max_x = min(width, x + radius + 1)
        min_y = max(0, y - radius)
        max_y = min(height, y + radius + 1)

        # Create coordinate grids for vectorized distance calculation
        xx, yy = np.meshgrid(
            np.arange(min_x, max_x), np.arange(min_y, max_y), indexing="ij"
        )

        # Calculate squared distance from the center (x, y)
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2

        # Create a mask for cells within the circular radius
        mask = dist_sq <= radius**2

        # Apply the mask to the sub-grid to mark as visited (1)
        self._grid[min_x:max_x, min_y:max_y][mask] = 1

        return is_new_step
