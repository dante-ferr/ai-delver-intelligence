from ..simulation import Simulation
from runtime.episode_trajectory import EpisodeTrajectory
from ai.config import config
from runtime.world_objects.entities import Entity
from runtime.episode_trajectory.snapshots import FrameSnapshot

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from level import Level
    from runtime.episode_trajectory import DelverAction


class ShowcaseSimulation(Simulation):
    """A simulation that records itself as an EpisodeTrajectory for showcase purposes."""

    def __init__(self, level: "Level"):
        super().__init__(level)

        self.episode_trajectory = EpisodeTrajectory(
            actions_per_second=config.ACTIONS_PER_SECOND, level_hash=level.to_hash()
        )

    def step(self, action: "DelverAction"):
        self.episode_trajectory.add_delver_action(action)

        ended, elapsed_time = super().step(action)

        if self.reached_goal:
            self.episode_trajectory.victorious = True

        return ended, elapsed_time

    def update(self, dt):
        super().update(dt)

        entities = self.world_objects_controller.get_world_objects_by_type(Entity)
        entities = cast(list[Entity], entities)
        frame_snapshot = FrameSnapshot()
        for entity in entities:
            frame_snapshot.add_entity(entity)
        self.episode_trajectory.add_frame_snapshot(frame_snapshot)
