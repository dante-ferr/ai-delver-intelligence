from runtime import Runtime
from runtime.episode_trajectory import EpisodeTrajectory
from ai.config import (
    FPS,
    FRAMES_PER_ACTION,
    FINISHED_REWARD,
    TURN_PENALTY_MULTIPLIER,
    MAX_FRAMES_PER_EPISODE,
    NOT_FINISHED_REWARD,
    FRAME_STEP_REWARD,
    TILE_EXPLORATION_REWARD,
)
from typing import TYPE_CHECKING
from ._exploration_grid import ExplorationGrid

if TYPE_CHECKING:
    from level import Level
    from runtime.episode_trajectory import DelverAction


DT = 1 / FPS * FRAMES_PER_ACTION


class Simulation(Runtime):

    def __init__(self, level: "Level"):
        super().__init__(level, render=False)

        self.elapsed_time = 0.0
        self.frame = 0

        self.total_reward = 0

        self.exploration_grid = ExplorationGrid(level.map.size[0], level.map.size[1])

        self.episode_trajectory = EpisodeTrajectory()
        self.last_action: "None | DelverAction" = None

    def step(self, action: "DelverAction"):
        """
        Handles an action from the delver on each step and updates the simulation.
        """

        self.episode_trajectory.add_delver_action(action)

        if action["move"]:
            self.delver.move(DT, action["move_angle"])

        self.last_action = action
        self.update(DT)

        reward = self._get_reward(action)
        self.total_reward += reward

        self.frame += 1

        if self.ended:
            self._end()

        return reward, self.ended, self.elapsed_time

    def _get_reward(self, action: "DelverAction"):
        reward = 0

        if self.reached_goal:
            reward += FINISHED_REWARD
        elif self.time_is_over:
            reward += NOT_FINISHED_REWARD

        if self.last_action and self.last_action["move"] and action["move"]:
            angle_diff = (
                action["move_angle"] - self.last_action["move_angle"] + 180
            ) % 360 - 180

            # Penalize for turning. The penalty is proportional to the change in angle.
            # Max penalty is 1.0 for a 180 degree turn.
            reward -= (abs(angle_diff) / 180.0) * TURN_PENALTY_MULTIPLIER

        reward += FRAME_STEP_REWARD

        new_explorated_tile = self.exploration_grid.step_on(self.delver.position)
        if new_explorated_tile:
            reward += TILE_EXPLORATION_REWARD

        return reward

    def _end(self):
        pass

    @property
    def ended(self):
        return self.reached_goal or self.time_is_over

    @property
    def reached_goal(self):
        return self.delver.check_collision(self.goal)

    @property
    def time_is_over(self):
        return self.frame >= MAX_FRAMES_PER_EPISODE

    def update(self, dt):
        super().update(dt)

        self.elapsed_time += dt
