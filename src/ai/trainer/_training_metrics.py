import tensorflow as tf
from tf_agents.metrics import tf_metrics
from typing import List, Any


class TrainingMetrics:
    """Helper class to encapsulate training metrics."""

    def __init__(self, batch_size: int):
        self.avg_return = tf_metrics.AverageReturnMetric(
            batch_size=batch_size, dtype=tf.float32
        )
        self.avg_episode_length = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=batch_size, dtype=tf.float32
        )
        self.num_episodes = tf_metrics.NumberOfEpisodes()

    def reset(self) -> None:
        self.avg_return.reset()
        self.avg_episode_length.reset()
        self.num_episodes.reset()

    @property
    def observers(self) -> List[Any]:
        return [self.avg_return, self.avg_episode_length, self.num_episodes]
