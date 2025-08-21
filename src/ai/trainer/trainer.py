import tensorflow as tf
from tf_agents.environments import parallel_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from ai.environments.level.environment import LevelEnvironment
import datetime
import functools
import logging
from ai.agents import PPOAgentFactory
from tensorflow.summary import create_file_writer  # type: ignore
from ai.config import config
from tf_agents.environments import tf_py_environment
from ._web_socket_observer import WebSocketObserver
from ._trainer_model_manager import TrainerModelManager
from ai.sessions.session_manager import session_manager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.sessions.session_manager import TrainingSession
    import asyncio


def _create_level_environment(env_id: int, level_json: dict, session_id: str):
    """Factory to create a LevelEnvironment instance in a subprocess."""
    return LevelEnvironment(env_id=env_id, level_json=level_json, session_id=session_id)


class Trainer:

    def __init__(
        self,
        session: "TrainingSession",
        loop: "asyncio.AbstractEventLoop",
        model_bytes: None | bytes = None,
    ):
        self.session = session
        self.loop = loop

        self.num_iteractions = (
            session.amount_of_episodes // config.COLLECT_STEPS_PER_ITERATION
        )

        self.model_bytes = model_bytes

        self._is_interrupted = False

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/train/" + current_time
        self.summary_writer = create_file_writer(log_dir)

        self.model_manager = TrainerModelManager(self)

    def setup_env_and_agent(self):
        constructors = [
            functools.partial(
                _create_level_environment,
                env_id=i,
                level_json=self.session.level_json,
                session_id=self.session.session_id,
            )
            for i in range(config.ENV_BATCH_SIZE)
        ]

        py_env = parallel_py_environment.ParallelPyEnvironment(
            constructors, start_serially=False
        )
        self.train_env = tf_py_environment.TFPyEnvironment(py_env)

        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=config.LEARNING_RATE, gamma=config.GAMMA
        ).get_agent()

        if self.model_bytes:
            logging.info(
                "Pre-trained model provided. Attempting to deserialize and load policy."
            )
            try:
                self.model_manager.load_serialized_model(self.model_bytes)
            except Exception as e:
                logging.error(
                    f"Failed to deserialize or load the model: {e}. Training will start with a new model."
                )
                # If loading fails, ensure we start from step 0.
                self.agent.train_step_counter.assign(0)
        else:
            # If no model is provided, start training from scratch.
            self.agent.train_step_counter.assign(0)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=config.REPLAY_BUFFER_CAPACITY,
        )

        self.avg_return_metric = tf_metrics.AverageReturnMetric(
            batch_size=self.train_env.batch_size
        )
        self.avg_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=self.train_env.batch_size
        )

        websocket_observer = WebSocketObserver(self.session.replay_queue, self.loop)

        self.driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[
                self.replay_buffer.add_batch,
                self.avg_return_metric,
                self.avg_episode_length_metric,
                websocket_observer,
            ],
            num_episodes=config.COLLECT_STEPS_PER_ITERATION,
        )

    def train(self):
        """Trains the agent until interrupted or the maximum number of iterations is reached. Returns the updated tensorflow model."""
        logging.info(f"Starting training for {self.num_iteractions} iterations...")
        self.driver.run = common.function(self.driver.run)

        with self.summary_writer.as_default():
            for iteration in range(self.num_iteractions):
                # Check for interruption before starting a new iteration.
                if self._is_interrupted:
                    logging.info("Training was interrupted by a user request.")
                    break

                # Collect a sequence of episodes.
                self.driver.run()

                # Train the agent with the collected experience.
                experience = self.replay_buffer.gather_all()
                loss_info = self.agent.train(experience)
                self.replay_buffer.clear()

                step = self.agent.train_step_counter
                if iteration % config.LOG_INTERVAL == 0:
                    logging.info(
                        f"Iteration {iteration}: Step = {step}, Loss = {loss_info.loss.numpy()}"
                    )
                    self._handle_tensorboard(loss_info, step)

        if not self._is_interrupted:
            logging.info("Training finished.")
        session_manager.delete_session(self.session.session_id)

    def interrupt_training(self):
        logging.info(
            "Interrupt signal received. Training will stop after the current iteration."
        )
        self._is_interrupted = True

    def _handle_tensorboard(self, loss_info, step):
        avg_return = self.avg_return_metric.result().numpy()
        tf.summary.scalar("average_return", avg_return, step=step)
        self.avg_return_metric.reset()

        avg_length = self.avg_episode_length_metric.result().numpy()
        tf.summary.scalar("average_episode_length", avg_length, step=step)
        self.avg_episode_length_metric.reset()

        tf.summary.scalar("loss", loss_info.loss, step=step)
