import tensorflow as tf
from tf_agents.environments import parallel_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.specs import tensor_spec
from ai.environments.level.environment import LevelEnvironment
import datetime
import functools
import logging
import gc
import time
from ai.agents import PPOAgentFactory
from tensorflow.summary import create_file_writer  # type: ignore
from ai.config import config
from tf_agents.environments import tf_py_environment
from ._web_socket_observer import WebSocketObserver
from ._trainer_model_manager import TrainerModelManager
from ai.sessions.session_manager import session_manager
from typing import TYPE_CHECKING
from tf_agents.policies import random_tf_policy

if TYPE_CHECKING:
    from ai.sessions.session_manager import TrainingSession
    import asyncio

BENCHMARK_MODE = False


def _create_level_environment(env_id: int, level_json: dict, session_id: str):
    return LevelEnvironment(env_id=env_id, level_json=level_json, session_id=session_id)


class Trainer:
    """
    Manages the training lifecycle of the RL agent.
    Handles environment creation, agent setup, replay buffer management,
    and the main training loop.
    """
    def __init__(
        self,
        session: "TrainingSession",
        loop: "asyncio.AbstractEventLoop",
        model_bytes: None | bytes = None,
    ):
        self.session = session
        self.loop = loop
        self.target_episodes = session.amount_of_episodes
        self.model_bytes = model_bytes
        self._is_interrupted = False

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/train/" + current_time
        self.summary_writer = create_file_writer(log_dir)
        self.model_manager = TrainerModelManager(self)

    def setup_env_and_agent(self):
        """Initializes all components required for training."""
        self._setup_environment()
        self._setup_agent()
        collect_policy, collect_data_spec = self._get_policy_and_spec()
        self._setup_replay_buffer(collect_data_spec)
        self._setup_metrics()
        self._setup_driver(collect_policy)

    def _setup_environment(self):
        """Creates a parallelized TFPyEnvironment."""
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
            constructors, start_serially=True
        )
        self.train_env = tf_py_environment.TFPyEnvironment(py_env)

    def _setup_agent(self):
        """Initializes the PPO agent and loads weights if provided."""
        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=config.LEARNING_RATE, gamma=config.GAMMA
        ).get_agent()

        if self.model_bytes:
            logging.info("Pre-trained model provided. Attempting to deserialize.")
            try:
                self.model_manager.load_serialized_model(self.model_bytes)
            except Exception as e:
                logging.error(f"Failed to load model: {e}. Starting fresh.")
                self.agent.train_step_counter.assign(0)
        else:
            self.agent.train_step_counter.assign(0)

    def _get_policy_and_spec(self):
        """Determines the collection policy and data spec based on benchmark mode."""
        if BENCHMARK_MODE:
            logging.warning("BENCHMARK MODE ACTIVE: Using Random Policy.")
            collect_policy = random_tf_policy.RandomTFPolicy(
                self.train_env.time_step_spec(), self.train_env.action_spec()
            )
            collect_data_spec = collect_policy.trajectory_spec
        else:
            collect_policy = self.agent.collect_policy
            collect_data_spec = self.agent.collect_data_spec
        return collect_policy, collect_data_spec

    def _setup_replay_buffer(self, collect_data_spec):
        """
        Sets up the replay buffer with selective float16 casting for observations
         to save memory while preserving precision for critical fields.
        """

        def _selective_cast(spec):
            if hasattr(spec, "dtype") and spec.dtype == tf.float32:
                # Don't cast BoundedTensorSpec (these are often important boundaries)
                if not isinstance(spec, tensor_spec.BoundedTensorSpec):
                    return tensor_spec.TensorSpec(
                        shape=spec.shape, dtype=tf.float16, name=spec.name
                    )
            return spec

        if hasattr(collect_data_spec, "observation"):
            from tf_agents.trajectories import trajectory

            observation_spec = tf.nest.map_structure(
                _selective_cast, collect_data_spec.observation
            )

            # Reconstruct the trajectory spec with casted observations
            collect_data_spec = collect_data_spec._replace(observation=observation_spec)
        else:
            collect_data_spec = tf.nest.map_structure(
                _selective_cast, collect_data_spec
            )

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=config.REPLAY_BUFFER_CAPACITY,
        )

    def _setup_metrics(self):
        """Initializes training metrics."""
        self.avg_return_metric = tf_metrics.AverageReturnMetric(
            batch_size=self.train_env.batch_size, dtype=tf.float32
        )
        self.avg_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=self.train_env.batch_size, dtype=tf.float32
        )
        self.num_episodes_metric = tf_metrics.NumberOfEpisodes()

    def _setup_driver(self, collect_policy):
        """Initializes the DynamicStepDriver for data collection."""
        observers = [
            self.replay_buffer.add_batch,
            self.avg_return_metric,
            self.avg_episode_length_metric,
            self.num_episodes_metric,
        ]

        if not BENCHMARK_MODE:
            observers.append(WebSocketObserver(self.session.replay_queue, self.loop))

        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            collect_policy,
            observers=observers,
            num_steps=config.COLLECT_STEPS_PER_ITERATION,
        )

    def train(self):
        """Main training loop."""
        logging.info(f"Starting training for target {self.target_episodes} episodes...")

        time_step = self.train_env.current_time_step()
        policy_state = self.driver.policy.get_initial_state(self.train_env.batch_size)
        self.driver.run = common.function(self.driver.run)

        iteration = 0
        with self.summary_writer.as_default():
            while self.num_episodes_metric.result().numpy() < self.target_episodes:
                if self._is_interrupted:
                    logging.info("Training interrupted by user.")
                    break

                try:
                    time_step, policy_state = self.driver.run(
                        time_step=time_step, policy_state=policy_state
                    )
                except Exception as e:
                    logging.error(f"Critical error during collection: {e}")
                    break

                self._run_training_step(iteration)

                self.replay_buffer.clear()
                if iteration % 10 == 0:
                    gc.collect()

                iteration += 1

        if not self._is_interrupted:
            logging.info(
                f"Training finished. Completed {self.num_episodes_metric.result().numpy()} episodes."
            )
        self._graceful_shutdown()

    def _run_training_step(self, iteration):
        """Performs a single gradient update using the collected experience."""
        if BENCHMARK_MODE:
            if iteration % config.LOG_INTERVAL == 0:
                logging.info(
                    f"Iter {iteration}: Benchmark Mode - Episodes: {self.num_episodes_metric.result().numpy()}"
                )
            return

        try:
            experience = self.replay_buffer.gather_all()
            loss_info = self.agent.train(experience)
            step = self.agent.train_step_counter

            if iteration % config.LOG_INTERVAL == 0:
                logging.info(
                    f"Iter {iteration}: Step={step.numpy()}, Loss={loss_info.loss.numpy():.4f}, Episodes={self.num_episodes_metric.result().numpy()}"
                )
                self._handle_tensorboard(loss_info, step)

        except tf.errors.ResourceExhaustedError:
            logging.warning(f"⚠️ OOM at Iteration {iteration}. Skipping step.")
            tf.compat.v1.reset_default_graph()
            gc.collect()
        except Exception as e:
            logging.error(f"Unexpected error during training: {e}")

    def _graceful_shutdown(self):
        """Cleans up resources and closes environments."""
        logging.info("Initiating graceful shutdown...")
        self.replay_buffer.clear()
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            self.train_env.close()
            logging.info("Environments closed.")
        except Exception as e:
            logging.warning(f"Error closing environments: {e}")
        time.sleep(3)
        session_manager.delete_session(self.session.session_id)
        logging.info("Session cleaned up.")

    def interrupt_training(self):
        """Signals the training loop to stop."""
        logging.info("Interrupt signal received.")
        self._is_interrupted = True

    def _handle_tensorboard(self, loss_info, step):
        """Logs metrics to TensorBoard."""
        tf.summary.scalar(
            "average_return", self.avg_return_metric.result().numpy(), step=step
        )
        self.avg_return_metric.reset()

        tf.summary.scalar(
            "average_episode_length",
            self.avg_episode_length_metric.result().numpy(),
            step=step,
        )
        self.avg_episode_length_metric.reset()

        tf.summary.scalar("loss", loss_info.loss, step=step)
        tf.summary.scalar(
            "total_episodes", self.num_episodes_metric.result().numpy(), step=step
        )
