from tf_agents.typing.types import NestedArraySpec
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, TYPE_CHECKING
import time
from functools import cached_property
from .simulation import Simulation
from runtime.episode_trajectory import DelverAction
from ._logger import LevelEnvironmentLogger
from ai.sessions import REGISTRY_LOCK, SESSION_REGISTRY
from level import Level

if TYPE_CHECKING:
    from ._delver_observation import DelverObservation

class LevelEnvironment(PyEnvironment):
    WINDOW_SIZE = 15

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
            self._initial_frame_count = self._last_frame_count

        self._restart_simulation()
        self._episodes = 0

        if self._env_id == 0:
            self._logger = LevelEnvironmentLogger()

        self._init_specs()
        self._episode_ended = False
        self._last_fps_time = time.time()
        self._start_time = time.time()

        self.fps = 0.0
        self.avg_fps = 0.0

    def _restart_simulation(self):
        self.simulation = Simulation(self._level)

    def _init_specs(self):
        self._action_spec = {
            "run": array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=2, name="run"
            ),
            "jump": array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=1, name="jump"
            ),
        }

        self._observation_spec = {
            # Local 15x15 grid centered on player. uint8 is efficient for IPC.
            "local_view": array_spec.ArraySpec(
                shape=(self.WINDOW_SIZE, self.WINDOW_SIZE),
                dtype=np.uint8,
                name="local_view",
            ),
            # [PlayerX, PlayerY, VelX, VelY, GoalX, GoalY] normalized
            "global_state": array_spec.ArraySpec(
                shape=(6,), dtype=np.float32, name="global_state"
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
        return ts.restart(self.observation)

    def _count_frame(self):
        with self._global_frame_lock:
            self._global_frame_counter.value += 1

    def _calculate_fps(self):
        with self._global_frame_lock:
            current_frame = self._global_frame_counter.value

        current_time = time.time()
        total_time = current_time - self._start_time
        total_frames = current_frame - self._initial_frame_count

        if total_time > 0:
            self.avg_fps = total_frames / total_time
        return self.avg_fps

    def _step(self, action):
        self._count_frame()
        if self._episode_ended:
            return self._reset()

        action_dict = self._get_dict_of_action(action)
        reward, self._episode_ended, _ = self.simulation.step(action_dict)

        if self._env_id == 0:
            self._logger.log_step(
                reward=reward,
                run=action_dict["run"],
                jump=action_dict["jump"],
                delver_position=self.delver_position,
                global_frame_count=self._global_frame_counter.value,
                simulation_frame=self.simulation.frame,
                fps=self._calculate_fps(),
            )

        return self._create_time_step(reward)

    def _get_dict_of_action(self, action) -> DelverAction:
        run_raw = int(action["run"])
        jump = bool(action["jump"])
        run = run_raw - 1
        return DelverAction(run=run, jump=jump)

    def _get_local_view(self):
        """Optimized slicing of the tilemap grid."""
        grid = self.platforms_grid
        h, w = grid.shape

        tx = int(self.delver_position[0] // self._level.map.tile_size[0])
        ty = int(self.delver_position[1] // self._level.map.tile_size[1])

        half = self.WINDOW_SIZE // 2

        # Calculate bounds with padding
        y_start, y_end = ty - half, ty + half + 1
        x_start, x_end = tx - half, tx + half + 1

        # Default to walls (1) for out of bounds
        view = np.ones((self.WINDOW_SIZE, self.WINDOW_SIZE), dtype=np.uint8)

        # Intersection between window and map
        grid_y_s, grid_y_e = max(0, y_start), min(h, y_end)
        grid_x_s, grid_x_e = max(0, x_start), min(w, x_end)

        view_y_s, view_x_s = max(0, -y_start), max(0, -x_start)
        view_y_e = view_y_s + (grid_y_e - grid_y_s)
        view_x_e = view_x_s + (grid_x_e - grid_x_s)

        if grid_y_e > grid_y_s and grid_x_e > grid_x_s:
            view[view_y_s:view_y_e, view_x_s:view_x_e] = grid[
                grid_y_s:grid_y_e, grid_x_s:grid_x_e
            ]

        return view

    def _create_time_step(self, reward):
        if self._episode_ended:
            obs = self.observation
            # Only populate replay JSON on termination to save bandwidth/CPU
            obs["replay_json"] = np.array(
                self.simulation.episode_trajectory.to_json(), dtype=str
            )
            return ts.termination(obs, reward)
        return ts.transition(self.observation, reward, 1.0)

    @property
    def observation(self) -> "DelverObservation":
        lvl_w = max(1, self._level.map.size[0] * self._level.map.tile_size[0])
        lvl_h = max(1, self._level.map.size[1] * self._level.map.tile_size[1])

        global_state = np.array(
            [
                self.delver_position[0] / lvl_w,
                self.delver_position[1] / lvl_h,
                self.simulation.delver.velocity[0] / 500.0,
                self.simulation.delver.velocity[1] / 1000.0,
                self.goal_position[0] / lvl_w,
                self.goal_position[1] / lvl_h,
            ],
            dtype=np.float32,
        )

        return {
            "local_view": self._get_local_view(),
            "global_state": global_state,
            "replay_json": np.array("", dtype=str),
        }

    @cached_property
    def platforms_grid(self):
        layer = self.simulation.tilemap.get_layer("platforms")
        return np.array(
            [[1 if cell is not None else 0 for cell in row] for row in layer.grid],
            dtype=np.uint8,
        )

    @property
    def delver_position(self):
        return self.simulation.delver.position

    @property
    def goal_position(self):
        return self.simulation.goal.position
