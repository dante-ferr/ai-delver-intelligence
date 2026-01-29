from runtime import Runtime
from ai.config import config
from ._exploration_grid import ExplorationGrid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from level import Level
    from runtime.episode_trajectory import DelverAction


DT = 1 / config.ACTIONS_PER_SECOND


class Simulation(Runtime):
    def __init__(self, level: "Level"):
        super().__init__(level, render=False)

        self.elapsed_time = 0.0
        self.frame = 0

        self.exploration_grid = ExplorationGrid(level.map.size[0], level.map.size[1])

        self.last_action: "None | DelverAction" = None

    def step(self, action: "DelverAction"):
        """
        Handles an action from the delver on each step and updates the simulation.
        """

        if action["run"] != 0:
            self.delver.run(DT, action["run"])

        if action["jump"]:
            self.delver.jump(DT)

        self.last_action = action
        self.update(DT)

        self.frame += 1

        return self.ended, self.elapsed_time

    @property
    def ended(self):
        return self.reached_goal or self.time_is_over

    @property
    def reached_goal(self):
        return self.delver.check_collision(self.goal)

    @property
    def time_is_over(self):
        return self.frame >= config.MAX_SECONDS_PER_EPISODE * config.ACTIONS_PER_SECOND

    def update(self, dt):
        super().update(dt)

        self.elapsed_time += dt
