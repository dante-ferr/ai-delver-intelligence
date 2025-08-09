import tensorflow as tf
from tf_agents.environments import parallel_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from ai.environments.level.environment import LevelEnvironment
import datetime
import functools
import dill
import logging
from ai.agents import PPOAgentFactory
from tensorflow.summary import create_file_writer  # type: ignore
from ai.config import *
from tf_agents.environments import tf_py_environment
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from level import Level
    from multiprocessing.managers import ValueProxy


def _create_level_environment(
    env_id: int,
    level_bytes: bytes,
    replay_queue,
    frame_counter: "ValueProxy",
    frame_lock,
):
    """Factory to create a LevelEnvironment instance in a subprocess."""
    return LevelEnvironment(
        env_id=env_id,
        level_bytes=level_bytes,
        replay_queue=replay_queue,
        frame_counter=frame_counter,
        frame_lock=frame_lock,
    )


class Trainer:

    def __init__(
        self,
        level: "Level",
        replay_queue,
        frame_counter: "ValueProxy",
        frame_lock,
    ):
        logging.info(f"Trainer __init__ started for level: {level}")
        self.level = level
        self.replay_queue = replay_queue
        self.global_frame_counter = frame_counter
        self.frame_lock = frame_lock

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/train/" + current_time
        self.summary_writer = create_file_writer(log_dir)

        self._setup_env_and_agent()
        logging.info("Trainer __init__ finished successfully.")

    def _setup_env_and_agent(self):
        constructors = [
            functools.partial(
                _create_level_environment,
                env_id=i,
                level_bytes=dill.dumps(self.level),
                replay_queue=self.replay_queue,
                frame_counter=self.global_frame_counter,
                frame_lock=self.frame_lock,
            )
            for i in range(ENV_BATCH_SIZE)
        ]

        py_env = parallel_py_environment.ParallelPyEnvironment(
            constructors, start_serially=False
        )
        self.train_env = tf_py_environment.TFPyEnvironment(py_env)

        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=LEARNING_RATE, gamma=GAMMA
        ).get_agent()
        self.agent.train_step_counter.assign(0)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=REPLAY_BUFFER_CAPACITY,
        )

        self.avg_return_metric = tf_metrics.AverageReturnMetric(
            batch_size=self.train_env.batch_size
        )
        self.avg_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=self.train_env.batch_size
        )

        self.driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[
                self.replay_buffer.add_batch,
                self.avg_return_metric,
                self.avg_episode_length_metric,
            ],
            num_episodes=COLLECT_STEPS_PER_ITERATION,
        )

    def train(self):
        logging.info(f"Starting training for {NUM_ITERATIONS} iterations...")
        self.driver.run = common.function(self.driver.run)

        with self.summary_writer.as_default():
            for iteration in range(NUM_ITERATIONS):
                self.driver.run()
                experience = self.replay_buffer.gather_all()
                loss_info = self.agent.train(experience)
                self.replay_buffer.clear()

                step = self.agent.train_step_counter
                if iteration % LOG_INTERVAL == 0:
                    logging.info(
                        f"Iteration {iteration}: Step = {step}, Loss = {loss_info.loss.numpy()}"
                    )
                    self._handle_tensorboard(loss_info, step)
        logging.info("Training finished.")

    def _handle_tensorboard(self, loss_info, step):
        avg_return = self.avg_return_metric.result().numpy()
        tf.summary.scalar("average_return", avg_return, step=step)
        self.avg_return_metric.reset()

        avg_length = self.avg_episode_length_metric.result().numpy()
        tf.summary.scalar("average_episode_length", avg_length, step=step)
        self.avg_episode_length_metric.reset()

        tf.summary.scalar("loss", loss_info.loss, step=step)
