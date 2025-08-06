from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArraySpec
import numpy as np
from typing import cast, Any, TYPE_CHECKING
import time
import math
from functools import cached_property
from multiprocessing import Manager
from .simulation import Simulation
from runtime.episode_trajectory import DelverAction
from level_holder import level_holder
from ._logger import LevelEnvironmentLogger

if TYPE_CHECKING:
    from .. import DelverObservation

SIMULATION_WS_URL = "ws://host.docker.internal:8000/ws/simulation"

manager = Manager()
global_frame_counter = manager.Value("i", 0)
frame_lock = manager.Lock()


class LevelEnvironment(PyEnvironment):
    def __init__(self, env_id: int = 0):
        self.env_id = env_id
        self.last_action: dict[str, Any] = {
            "move": 0.0,
            "move_angle_sin": 0.0,
            "move_angle_cos": 0.0,
        }
        self._restart_simulation()
        self.episodes = 0

        # self.pathfinder = Pathfinder(self)
        if self.env_id == 0:
            self.logger = LevelEnvironmentLogger()

        self._init_specs()

        self.episode_ended = False
        self.last_fps_time = time.time()

        with frame_lock:
            self.last_frame_count = global_frame_counter.value
        self.fps = 0.0

    def _restart_simulation(self):
        self.simulation = Simulation(level_holder.level)

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

        self.observation_shape = (3,)
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
            # "path_direction": array_spec.ArraySpec(
            #     shape=(2,), dtype=np.float32, name="path_direction"
            # ),
        }

    def _reset(self):
        self.episodes += 1
        self.episode_ended = False
        self._restart_simulation()

        if self.env_id == 0:
            self.logger.log_episode_start(self.episodes)

        return ts.restart(self.observation)

    def _count_frame(self):
        with frame_lock:
            global_frame_counter.value += 1

    def _calculate_fps(self):
        with frame_lock:
            current_frame = global_frame_counter.value
        current_time = time.time()

        time_delta = current_time - self.last_fps_time
        frame_delta = current_frame - self.last_frame_count

        fps = frame_delta / time_delta
        self.last_fps_time = current_time
        self.last_frame_count = current_frame

        return fps

    def _step(self, action):
        self._count_frame()

        if self.episode_ended:
            return self._reset()

        action_dict = self._get_dict_of_action(action)
        reward, self.episode_ended, elapsed_time = self.simulation.step(action_dict)

        if self.env_id == 0:
            self.logger.log_step(
                reward=reward,
                move=action_dict["move"],
                move_angle=action_dict["move_angle"],
                delver_position=self.observation["delver_position"],
                global_frame_count=global_frame_counter.value,
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

    # tf_agents.typing.types.NestedArraySpec is a union that includes tf_agents.types.ArraySpec. So I suppose it's safe to cast it to bounded arrays, because they extend ArraySpec.
    def action_spec(self):
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        return cast(NestedArraySpec, self._observation_spec)

    @property
    def observation(self):
        walls_layer = self.walls_grid.astype(np.float32)

        observation: "DelverObservation" = {
            "walls": np.array(walls_layer, dtype=np.float32),
            "delver_position": np.array([*self.delver_position], dtype=np.float32),
            "goal_position": np.array([*self.goal_position], dtype=np.float32),
        }
        return observation

    @cached_property
    def walls_grid(self):
        walls_grid = self.simulation.tilemap.get_layer("walls").grid
        walls_grid_presence = np.array(
            [[1 if cell is not None else 0 for cell in row] for row in walls_grid],
            dtype=np.uint8,
        )
        return walls_grid_presence

    @property
    def delver_position(self):
        return self.simulation.delver.position

    @property
    def goal_position(self):
        return self.simulation.goal.position
