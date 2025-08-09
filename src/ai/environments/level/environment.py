from tf_agents.typing.types import NestedArraySpec
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, Any, TYPE_CHECKING
import time
import math
import dill
from functools import cached_property

from .simulation import Simulation
from runtime.episode_trajectory import DelverAction
from ._logger import LevelEnvironmentLogger

if TYPE_CHECKING:
    from level import Level
    from ._delver_observation import DelverObservation
    from multiprocessing.managers import ValueProxy

class LevelEnvironment(PyEnvironment):

    def __init__(
        self,
        env_id: int,
        level_bytes: bytes,
        replay_queue,
        frame_counter: "ValueProxy",
        frame_lock,
    ):
        self.env_id = env_id
        self.level: "Level" = dill.loads(level_bytes)
        self.replay_queue = replay_queue

        # Store shared objects as instance attributes
        self.global_frame_counter = frame_counter
        self.frame_lock = frame_lock

        self._restart_simulation()
        self.episodes = 0

        if self.env_id == 0:
            self.logger = LevelEnvironmentLogger()

        self._init_specs()
        self.episode_ended = False
        self.last_fps_time = time.time()

        with self.frame_lock:
            self.last_frame_count = self.global_frame_counter.value
        self.fps = 0.0

    def _restart_simulation(self):
        self.simulation = Simulation(self.level)

    def _init_specs(self):
        self._action_spec = {
            "move": array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="move"
            ),
            "move_angle_cos": array_spec.BoundedArraySpec(
                (), dtype=np.float32, minimum=-1.0, maximum=1.0, name="move_angle_cos"
            ),
            "move_angle_sin": array_spec.BoundedArraySpec(
                (), dtype=np.float32, minimum=-1.0, maximum=1.0, name="move_angle_sin"
            ),
        }
        self._observation_spec = {
            "walls": array_spec.ArraySpec(
                shape=self.walls_grid.shape, dtype=np.float32, name="walls"
            ),
            "delver_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="delver_position"
            ),
            "goal_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="goal_position"
            ),
        }

    def action_spec(self):
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        return cast(NestedArraySpec, self._observation_spec)

    def _reset(self):
        self.episodes += 1
        self.episode_ended = False
        self._restart_simulation()
        if self.env_id == 0:
            self.logger.log_episode_start(self.episodes)
        return ts.restart(self.observation)

    def _count_frame(self):
        with self.frame_lock:
            self.global_frame_counter.value += 1

    def _calculate_fps(self):
        with self.frame_lock:
            current_frame = self.global_frame_counter.value
        current_time = time.time()
        time_delta = current_time - self.last_fps_time
        frame_delta = current_frame - self.last_frame_count
        if time_delta == 0:
            return self.fps
        self.fps = frame_delta / time_delta
        self.last_fps_time = current_time
        self.last_frame_count = current_frame
        return self.fps

    def _step(self, action):
        self._count_frame()
        if self.episode_ended:
            if self.replay_queue:
                self.replay_queue.put(self.simulation.episode_trajectory.to_json())
            return self._reset()

        action_dict = self._get_dict_of_action(action)
        reward, self.episode_ended, _ = self.simulation.step(action_dict)

        if self.env_id == 0:
            self.logger.log_step(
                reward=reward,
                move=action_dict["move"],
                move_angle=action_dict["move_angle"],
                delver_position=self.observation["delver_position"],
                global_frame_count=self.global_frame_counter.value,
                simulation_frame=self.simulation.frame,
                fps=self._calculate_fps(),
            )

        return self._create_time_step(reward)

    def _get_dict_of_action(self, action):
        move_angle_rad = math.atan2(action["move_angle_sin"], action["move_angle_cos"])
        return DelverAction(
            move=False if round(float(action["move"])) == 0 else True,
            move_angle=float(math.degrees(move_angle_rad)),
        )

    def _create_time_step(self, reward):
        if self.episode_ended:
            if self.env_id == 0:
                self.logger.log_episode_end(
                    episode=self.episodes, reward=self.simulation.total_reward
                )
            return ts.termination(self.observation, reward)
        return ts.transition(self.observation, reward, 1.0)

    @property
    def observation(self) -> "DelverObservation":
        walls_layer = self.walls_grid.astype(np.float32)
        return {
            "walls": np.array(walls_layer, dtype=np.float32),
            "delver_position": np.array([*self.delver_position], dtype=np.float32),
            "goal_position": np.array([*self.goal_position], dtype=np.float32),
        }

    @cached_property
    def walls_grid(self):
        walls_grid = self.simulation.tilemap.get_layer("walls").grid
        return np.array(
            [[1 if cell is not None else 0 for cell in row] for row in walls_grid],
            dtype=np.uint8,
        )

    @property
    def delver_position(self):
        return self.simulation.delver.position

    @property
    def goal_position(self):
        return self.simulation.goal.position
