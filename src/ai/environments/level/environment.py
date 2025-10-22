from tf_agents.typing.types import NestedArraySpec
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, TYPE_CHECKING
import time
import math
import dill
from functools import cached_property
from .simulation import Simulation
from runtime.episode_trajectory import DelverAction
from ._logger import LevelEnvironmentLogger
from ai.sessions import REGISTRY_LOCK, SESSION_REGISTRY
from level import Level

if TYPE_CHECKING:
    from ._delver_observation import DelverObservation

class LevelEnvironment(PyEnvironment):

    def __init__(self, env_id: int, level_json: dict, session_id: str):
        self._env_id = env_id
        self._level = Level.from_dict(level_json)
        self._session_id = session_id

        with REGISTRY_LOCK:
            session_objects = SESSION_REGISTRY[self._session_id]

        self._global_frame_counter = session_objects["frame_counter"]
        self._global_frame_lock = session_objects["frame_lock"]
        with self._global_frame_lock:
            self._last_frame_count = self._global_frame_counter.value

        self._restart_simulation()
        self._episodes = 0

        if self._env_id == 0:
            self._logger = LevelEnvironmentLogger()

        self._init_specs()
        self._episode_ended = False
        self._last_fps_time = time.time()

        self.fps = 0.0

    def _restart_simulation(self):
        self.simulation = Simulation(self._level)

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
            "platforms": array_spec.ArraySpec(
                shape=self.platforms_grid.shape, dtype=np.float32, name="platforms"
            ),
            "delver_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="delver_position"
            ),
            "goal_position": array_spec.ArraySpec(
                shape=(2,), dtype=np.float32, name="goal_position"
            ),
            "replay_json": array_spec.ArraySpec(
                shape=(), dtype=str, name="replay_json"
            ),
        }

    def action_spec(self):
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        return cast(NestedArraySpec, self._observation_spec)

    def _reset(self):
        self._episodes += 1
        self._episode_ended = False
        self._restart_simulation()
        if self._env_id == 0:
            self._logger.log_episode_start()
        return ts.restart(self.observation)

    def _count_frame(self):
        with self._global_frame_lock:
            self._global_frame_counter.value += 1

    def _calculate_fps(self):
        with self._global_frame_lock:
            current_frame = self._global_frame_counter.value

        current_time = time.time()
        time_delta = current_time - self._last_fps_time
        frame_delta = current_frame - self._last_frame_count
        if time_delta == 0:
            return self.fps

        self.fps = frame_delta / time_delta
        self._last_fps_time = current_time
        self._last_frame_count = current_frame
        return self.fps

    def _step(self, action):
        self._count_frame()
        if self._episode_ended:
            return self._reset()

        action_dict = self._get_dict_of_action(action)
        reward, self._episode_ended, _ = self.simulation.step(action_dict)

        if self._env_id == 0:
            self._logger.log_step(
                reward=reward,
                move=action_dict["move"],
                move_angle=action_dict["move_angle"],
                delver_position=self.observation["delver_position"],
                global_frame_count=self._global_frame_counter.value,
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
        if self._episode_ended:
            if self._env_id == 0:
                self._logger.log_episode_end(reward=self.simulation.total_reward)
            return ts.termination(self.observation, reward)
        return ts.transition(self.observation, reward, 1.0)

    @property
    def observation(self) -> "DelverObservation":
        replay_data_str = ""

        if self._episode_ended:
            replay_data_str = self.simulation.episode_trajectory.to_json()

        platforms_layer = self.platforms_grid.astype(np.float32)

        return {
            "platforms": np.array(platforms_layer, dtype=np.float32),
            "delver_position": np.array([*self.delver_position], dtype=np.float32),
            "goal_position": np.array([*self.goal_position], dtype=np.float32),
            "replay_json": np.array(replay_data_str, dtype=str),
        }

    @cached_property
    def platforms_grid(self):
        platforms_grid = self.simulation.tilemap.get_layer("platforms").grid
        return np.array(
            [[1 if cell is not None else 0 for cell in row] for row in platforms_grid],
            dtype=np.uint8,
        )

    @property
    def delver_position(self):
        return self.simulation.delver.position

    @property
    def goal_position(self):
        return self.simulation.goal.position
