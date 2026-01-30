from tf_agents.typing.types import NestedArraySpec
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from typing import cast, TYPE_CHECKING
import time
from functools import cached_property
from .simulation import Simulation
from .simulation.showcase_simulation import ShowcaseSimulation
from runtime.episode_trajectory import DelverAction
from ._logger import LevelEnvironmentLogger
from ._reward_calculator import RewardCalculator
from ai.sessions import REGISTRY_LOCK, SESSION_REGISTRY
from level import Level
from ._dijkstra_grid import DijkstraGrid
from ai.config import config

if TYPE_CHECKING:
    from ._delver_observation import DelverObservation


class LevelEnvironment(PyEnvironment):
    """
    A custom TF-Agents environment that wraps the Delver simulation.
    Supports both 'Training' (lightweight) and 'Showcase' (recording) modes.
    """
    WINDOW_SIZE = 15

    def __init__(
        self, env_id: int, level_json: dict, session_id: str, is_showcase: bool = False
    ):
        super().__init__()

        self._env_id = env_id
        self._level_json = level_json
        self._session_id = session_id
        self._is_showcase = is_showcase

        self._init_level_and_simulation()
        self._init_reward_system()

        # Showcase environments don't need global frame tracking or loggers
        if not self._is_showcase:
            self._init_global_frame_tracking(session_id)
            self._init_logger(env_id)
        else:
            self._global_frame_counter = None
            self._logger = None

        self._init_specs_and_types()
        self._init_runtime_metrics()

        self._reward_spec = array_spec.ArraySpec(
            shape=(), dtype=np.float32, name="reward"
        )
        self._discount_spec = array_spec.ArraySpec(
            shape=(), dtype=np.float32, name="discount"
        )

    def _init_level_and_simulation(self):
        """Initializes the simulation. Uses ShowcaseSimulation if is_showcase is True."""
        self._level = Level.from_dict(self._level_json)

        if self._is_showcase:
            self.simulation = ShowcaseSimulation(self._level)
        else:
            self.simulation = Simulation(self._level)

    def _init_reward_system(self):
        """Sets up the Dijkstra grid and the reward calculator."""
        # Instantiate Dijkstra Map locally.
        # Computing BFS for tilemaps is extremely fast (<1ms), so no need for complex JSON serialization overhead.
        tile_w, tile_h = self._level.map.tile_size
        goal_tx = int(self.goal_position[0] // tile_w)
        goal_ty = int(self.goal_position[1] // tile_h)

        self.dijkstra = DijkstraGrid(self._level.map.tilemap, (goal_tx, goal_ty))
        self.reward_calculator = RewardCalculator(self._level, self.dijkstra)
        self.reward_calculator.reset(self.simulation)

    def _init_global_frame_tracking(self, session_id: str):
        """Initializes shared global frame counters for synchronization."""
        with REGISTRY_LOCK:
            session_objects = SESSION_REGISTRY[session_id]

        self._global_frame_counter = session_objects["frame_counter"]
        self._global_frame_lock = session_objects["frame_lock"]
        with self._global_frame_lock:
            self._last_frame_count = self._global_frame_counter.value
            self._initial_frame_count = self._last_frame_count

    def _init_logger(self, env_id: int):
        """Initializes the episode counter and the logger for environment 0."""
        self._episodes = 0
        self._logger = None  # Ensure attribute exists for all envs
        if env_id == 0:
            self._logger = LevelEnvironmentLogger()

    def _init_specs_and_types(self):
        """Defines the action, observation, reward, and discount specifications."""
        self._init_specs()

    def _init_runtime_metrics(self):
        """Initializes variables for tracking episode state and performance metrics."""
        self._episode_ended = False
        self._start_time = time.time()
        self.fps = 0.0
        self.avg_fps = 0.0

    def _restart_simulation(self):
        """Re-initializes the simulation state for a new episode."""
        # Re-instantiate based on mode
        if self._is_showcase:
            self.simulation = ShowcaseSimulation(self._level)
        else:
            self.simulation = Simulation(self._level)

        self.reward_calculator.reset(self.simulation)

    def _init_specs(self):
        """Defines the action and observation schemas for the environment."""
        self._action_spec = {
            "run": array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=2, name="run"
            ),
            "jump": array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=1, name="jump"
            ),
        }

        self._observation_spec = {
            "local_view": array_spec.ArraySpec(
                shape=(self.WINDOW_SIZE, self.WINDOW_SIZE),
                dtype=np.uint8,
                name="local_view",
            ),
            "global_state": array_spec.ArraySpec(
                shape=(len(self.observation["global_state"]),),
                dtype=np.float32,
                name="global_state",
            ),
            "replay_json": array_spec.ArraySpec(
                shape=(), dtype=str, name="replay_json"
            ),
        }

    def action_spec(self):
        """Returns the action specification."""
        return cast(NestedArraySpec, self._action_spec)

    def observation_spec(self):
        """Returns the observation specification."""
        return cast(NestedArraySpec, self._observation_spec)

    def _reset(self):
        if not self._is_showcase:
            self._episodes += 1

        self._episode_ended = False
        self._restart_simulation()

        time_step = ts.restart(self.observation)
        return time_step._replace(
            reward=np.array(0.0, dtype=np.float32),
            discount=np.array(1.0, dtype=np.float32),
        )

    def _count_frame(self):
        if self._global_frame_counter:
            with self._global_frame_lock:
                self._global_frame_counter.value += 1

    def _calculate_fps(self):
        if not self._global_frame_counter:
            return 0.0

        with self._global_frame_lock:
            current_frame = self._global_frame_counter.value

        current_time = time.time()
        total_time = current_time - self._start_time
        total_frames = current_frame - self._initial_frame_count

        if total_time > 0:
            self.avg_fps = total_frames / total_time
        return self.avg_fps

    def _step(self, action):
        """Advances the simulation by one step based on the provided action."""
        self._count_frame()
        if self._episode_ended:
            return self._reset()

        action_dict = self._get_dict_of_action(action)
        self._episode_ended, _ = self.simulation.step(action_dict)

        reward = self.reward_calculator.calculate_reward(self.simulation, action_dict)
        reward = np.array(reward, dtype=np.float32)

        # Updated check: safe because self._logger is always initialized in _init_logger
        if self._logger and self._env_id == 0:
            human_readable_reward = float(reward) / config.REWARD_SCALE_FACTOR

            self._logger.log_step(
                reward=human_readable_reward,
                global_frame_count=(
                    self._global_frame_counter.value
                    if self._global_frame_counter
                    else 0
                ),
                simulation_frame=self.simulation.frame,
                fps=self._calculate_fps(),
            )

        return self._create_time_step(reward)

    def _get_dict_of_action(self, action) -> DelverAction:
        """Converts the RL action tensor into a simulation-compatible DelverAction."""
        run_raw = int(action["run"])
        jump = bool(action["jump"])
        run = run_raw - 1
        return DelverAction(run=run, jump=jump)

    def _get_local_view(self):
        """Extracts a cropped grid centered on the agent's current position."""
        grid = self.platforms_grid
        h, w = grid.shape

        tx = int(self.delver_position[0] // self._level.map.tile_size[0])
        ty = int(self.delver_position[1] // self._level.map.tile_size[1])

        half = self.WINDOW_SIZE // 2

        y_start, y_end = ty - half, ty + half + 1
        x_start, x_end = tx - half, tx + half + 1

        view = np.ones((self.WINDOW_SIZE, self.WINDOW_SIZE), dtype=np.uint8)

        # Calculate valid intersection between the requested window and the actual grid
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
        """Wraps the current observation and reward into a TF-Agents TimeStep."""
        if self._episode_ended:
            obs = self.observation

            # Logic: Only populate the JSON if we are in Showcase mode
            # This saves massive CPU/Memory during training
            replay_json_str = ""
            if self._is_showcase:
                self.simulation = cast(ShowcaseSimulation, self.simulation)
                replay_json_str = self.simulation.episode_trajectory.to_json()

            obs["replay_json"] = np.array(replay_json_str, dtype=str)

            time_step = ts.termination(obs, reward)
            return time_step._replace(discount=np.array(0.0, dtype=np.float32))

        return ts.transition(self.observation, reward, np.array(1.0, dtype=np.float32))

    @property
    def observation(self) -> "DelverObservation":
        """Constructs the current observation dictionary."""
        delver = self.simulation.delver
        goal = self.simulation.goal

        norm_goal_vector = (
            (goal.position[0] - delver.position[0])
            / config.DELVER_GOAL_DISTANCE_NORM[0],
            (goal.position[1] - delver.position[1])
            / config.DELVER_GOAL_DISTANCE_NORM[1],
        )
        norm_delver_velocity = (
            delver.velocity[0] / delver.MAX_SPEED[0],
            delver.velocity[1] / delver.MAX_SPEED[1],
        )
        # It's important to let the intelligence know the Delver's offset on the grid in order to make
        # the local view observation precise, as it's based on the grid instead of the actual pixel position.
        norm_delver_offset = (
            (delver.position[0] % config.TILE_WIDTH) / config.TILE_WIDTH,
            (delver.position[1] % config.TILE_HEIGHT) / config.TILE_HEIGHT,
        )

        global_state = np.array(
            [
                *norm_goal_vector,
                *norm_delver_velocity,
                *norm_delver_offset,
                # Passing the is_on_ground data is good because the intelligence won't need to figure out
                # whether the delver is on the ground or not by the local view and delver offset (it's a
                # challenging correlation).
                float(self.simulation.delver.is_on_ground),
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
        """Returns a binary numpy array representing the platform layer of the level."""
        layer = self.simulation.tilemap.get_layer("platforms")
        return np.array(
            [[1 if cell is not None else 0 for cell in row] for row in layer.grid],
            dtype=np.uint8,
        )

    @property
    def delver_position(self):
        """Current position of the agent (Pixels)."""
        return self.simulation.delver.position

    @property
    def goal_position(self):
        """Position of the level goal (Pixels)."""
        return self.simulation.goal.position
