import tensorflow as tf
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import trajectory
from collections import deque
import numpy as np
from ai.config import config
from typing import List, Any

class TrainingMetrics:
    """
    Encapsulates training metrics and dynamic graduation logic (Success Rate & Plateau).
    Uses a custom observer wrapped in tf.py_function to track Python-side statistics
    during graph execution.
    """

    def __init__(self, batch_size: int, window_size: int = 100):
        self._num_episodes = tf_metrics.NumberOfEpisodes()
        self._env_steps = tf_metrics.EnvironmentSteps()
        self._avg_return = tf_metrics.AverageReturnMetric(
            batch_size=batch_size, dtype=tf.float32
        )
        self._avg_episode_length = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=batch_size, dtype=tf.float32
        )

        self._window_size = window_size

        self._recent_binary_successes: deque = deque(maxlen=window_size)
        self._recent_returns: deque = deque(maxlen=window_size)

    @property
    def num_episodes(self):
        return self._num_episodes.result().numpy()  # type: ignore

    @property
    def avg_return(self):
        return self._avg_return.result().numpy()

    def reset_avg_return(self):
        self._avg_return.reset()

    def reset(self) -> None:
        self._num_episodes.reset()
        self._env_steps.reset()
        self._avg_return.reset()
        self._avg_episode_length.reset()
        self._recent_binary_successes.clear()
        self._recent_returns.clear()

    @property
    def observers(self) -> List[Any]:
        return [
            self._num_episodes,
            self._env_steps,
            self._avg_return,
            self._avg_episode_length,
            self._custom_observer,
        ]

    def _update_internal_stats(self, is_last_batch, reward_batch):
        """
        Python function executed eagerly to update deques.
        Iterates through the batch to capture all completed episodes.
        """
        for i in range(len(is_last_batch)):
            if is_last_batch[i]:

                reward = reward_batch[i]
                is_success = 1.0 if reward >= config.DYNAMIC_TARGET_REWARD else 0.0

                self._recent_binary_successes.append(is_success)
                self._recent_returns.append(reward)

        return np.float32(0)

    def _custom_observer(self, traj: trajectory.Trajectory):
        """
        TF Op wrapper that calls the python update function.
        """
        return tf.py_function(
            func=self._update_internal_stats,
            inp=[traj.is_last(), traj.reward],
            Tout=tf.float32,
        )

    def get_binary_success_rate(self) -> float:
        if not self._recent_binary_successes:
            return 0.0
        return sum(self._recent_binary_successes) / len(self._recent_binary_successes)

    def is_plateaued(self, tolerance: float = 0.01, min_episodes: int = 50) -> bool:
        """
        Checks if the agent has stopped learning by comparing the first half
        of the window to the second half.
        """
        if len(self._recent_returns) < min_episodes:
            return False

        # Split window
        half = len(self._recent_returns) // 2
        first_half = list(self._recent_returns)[:half]
        second_half = list(self._recent_returns)[half:]

        avg_1 = np.mean(first_half)
        avg_2 = np.mean(second_half)

        if abs(avg_1) < 1e-6:
            return False

        improvement = float((avg_2 - avg_1) / abs(avg_1))

        return improvement < tolerance
