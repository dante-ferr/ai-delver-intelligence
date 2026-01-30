from ai.config import config
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .simulation import Simulation
    from runtime.episode_trajectory import DelverAction
    from ._dijkstra_grid import DijkstraGrid
    from level import Level


class RewardCalculator:
    """
    Encapsulates all reward calculation logic for the environment.
    """

    def __init__(self, level: "Level", dijkstra_grid: "DijkstraGrid"):
        self._level = level
        self.dijkstra_grid = dijkstra_grid
        self.last_distance: float = 0.0
        self.last_delver_x: float = 0

    def reset(self, simulation: "Simulation"):
        """Resets the internal state for a new episode."""
        self.last_distance = self._get_current_dijkstra_distance(simulation)

    def _get_current_dijkstra_distance(self, simulation: "Simulation") -> float:
        """Helper to get current distance from the goal in tile coordinates."""
        tile_w, tile_h = self._level.map.tile_size

        # Convert Pixel Position -> Tile Position
        curr_tx = simulation.delver.position[0] / tile_w
        curr_ty = simulation.delver.position[1] / tile_h

        return self.dijkstra_grid.get_distance(curr_tx, curr_ty)

    def calculate_reward(
        self, simulation: "Simulation", action: "DelverAction"
    ) -> float:
        """Calculates the total reward for the current simulation step."""
        reward = 0.0

        if simulation.reached_goal:
            reward += config.FINISHED_REWARD
        elif simulation.time_is_over:
            reward += config.NOT_FINISHED_REWARD

        # Turn Penalty (Change of direction)
        if (
            simulation.last_action
            and simulation.last_action["run"] != 0
            and action["run"] != 0
            and simulation.last_action["run"] != action["run"]
        ):
            reward += config.TURN_REWARD

        # Jump Cost (Intent based)
        if action["jump"]:
            reward += config.JUMP_REWARD

        # Living Cost
        reward += config.FRAME_STEP_REWARD

        # Exploration Reward
        if simulation.exploration_grid.step_on(simulation.delver.position, 1):
            reward += config.TILE_EXPLORATION_REWARD

        # Dijkstra Distance Reward (The Compass)
        if config.GOAL_DISTANCE_REWARD_SCALE != 0.0:
            current_dist = self._get_current_dijkstra_distance(simulation)

            # Ensure valid distances before calculating delta
            if current_dist != -1.0 and self.last_distance != -1.0:
                # Delta is positive if we got closer (last > current)
                delta = self.last_distance - current_dist
                reward += delta * config.GOAL_DISTANCE_REWARD_SCALE

            if current_dist != -1.0:
                self.last_distance = current_dist

        # Wall Hugging penalty
        current_delver_x = simulation.delver.position[0]
        dx = abs(current_delver_x - self.last_delver_x)
        if (
            action["run"] != 0  # -1 is left and 1 is right, so 0 is not moving
            and dx < 0.001  # Barely moving, which we can assume is a wall hug
            and simulation.delver.is_on_ground
        ):
            reward += config.WALL_HUGGING_REWARD

        self.last_delver_x = current_delver_x

        return reward
