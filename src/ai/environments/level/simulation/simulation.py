from runtime import Runtime
from runtime.episode_trajectory import EpisodeTrajectory
from runtime.episode_trajectory.snapshots import FrameSnapshot
from ai.config import config
from ._exploration_grid import ExplorationGrid
from runtime.world_objects.entities import Entity
from typing import TYPE_CHECKING, cast

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

        self.episode_trajectory = EpisodeTrajectory(
            actions_per_second=config.ACTIONS_PER_SECOND
        )
        self.last_action: "None | DelverAction" = None

    def step(self, action: "DelverAction"):
        """
        Handles an action from the delver on each step and updates the simulation.
        """

        self.episode_trajectory.add_delver_action(action)

        if action["run"] != 0:
            self.delver.run(DT, action["run"])

        if action["jump"]:
            self.delver.jump(DT)

        self.last_action = action
        self.update(DT)

        self.frame += 1

        if self.reached_goal:
            self.episode_trajectory.victorious = True

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

        entities = self.world_objects_controller.get_world_objects_by_type(Entity)
        entities = cast(list[Entity], entities)
        frame_snapshot = FrameSnapshot()
        for entity in entities:
            frame_snapshot.add_entity(entity)
        self.episode_trajectory.add_frame_snapshot(frame_snapshot)

        self.elapsed_time += dt
