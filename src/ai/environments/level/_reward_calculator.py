from ai.config import config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation
    from runtime.episode_trajectory import DelverAction
    from ._dijkstra_grid import DijkstraGrid
    from level import Level


class RewardCalculator:
    """
    Encapsulates all reward calculation logic for the environment.
    This class is responsible for calculating the reward at each step of the simulation,
    including goal achievement, penalties, exploration, and distance-based rewards.
    """

    def __init__(self, level: "Level", dijkstra_grid: "DijkstraGrid"):
        self._level = level
        self.dijkstra_grid = dijkstra_grid
        self.last_distance: float = 0.0

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

        if (
            simulation.last_action
            and simulation.last_action["run"] != 0
            and action["run"] != 0
            and simulation.last_action["run"] != action["run"]
        ):
            reward += config.TURN_PENALTY

        if action["jump"]:
            reward += config.JUMP_REWARD

        reward += config.FRAME_STEP_REWARD

        if simulation.exploration_grid.step_on(simulation.delver.position, 1):
            reward += config.TILE_EXPLORATION_REWARD

        if config.DISTANCE_REWARD_SCALE != 0.0:
            current_dist = self._get_current_dijkstra_distance(simulation)
            if current_dist != -1.0 and self.last_distance != -1.0:
                delta = self.last_distance - current_dist
                reward += delta * config.DISTANCE_REWARD_SCALE

            if current_dist != -1.0:
                self.last_distance = current_dist

        return reward
