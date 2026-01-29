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
from tensorflow.summary import create_file_writer
from ai.config import config
from tf_agents.environments import tf_py_environment
from ._trainer_model_manager import TrainerModelManager
from ai.sessions.session_manager import session_manager
from ._evaluator import Evaluator
from typing import TYPE_CHECKING
from tf_agents.policies import random_tf_policy

if TYPE_CHECKING:
    from ai.sessions.session_manager import TrainingSession
    import asyncio

BENCHMARK_MODE = False


def _create_level_environment(env_id: int, level_json: dict, session_id: str):
    """
    Helper function to create a LevelEnvironment instance.
    For training, we force is_showcase=False to save resources.
    """
    return LevelEnvironment(
        env_id=env_id, level_json=level_json, session_id=session_id, is_showcase=False
    )


class Trainer:
    """
    Manages the training lifecycle of the RL agent.
    Orchestrates the loop between the 'Dojo' (Training Batches) and the 'Stage' (Showcase).
    """

    def __init__(
        self,
        session: "TrainingSession",
        loop: "asyncio.AbstractEventLoop",
        model_bytes: None | bytes = None,
    ):
        self.session = session
        self.loop = loop

        # We prioritize Cycle Count logic to ensure consistent showcases
        self.training_cycles = session.amount_of_cycles
        self.episodes_per_cycle = session.episodes_per_cycle

        self.model_bytes = model_bytes
        self._is_interrupted = False

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/train/" + current_time
        self.summary_writer = create_file_writer(log_dir)
        self.model_manager = TrainerModelManager(self)

        # Initialize the Evaluator (The Stage) for periodic showcases
        self.evaluator = Evaluator(session.level_json, session.session_id)

        # Placeholder for the optimized training function
        self._train_fn = None

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

        # Optimize the train function to handle variable batch sizes without retracing
        self._train_fn = common.function(self.agent.train, reduce_retracing=True)

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
            observation_spec = tf.nest.map_structure(
                _selective_cast, collect_data_spec.observation
            )
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
        """Initializes the DynamicStepDriver. Note: NO WebSocketObserver here."""
        observers = [
            self.replay_buffer.add_batch,
            self.avg_return_metric,
            self.avg_episode_length_metric,
            self.num_episodes_metric,
        ]

        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            collect_policy,
            observers=observers,
            num_steps=config.COLLECT_STEPS_PER_ITERATION,
        )

    def train(self):
        """Main training loop driven by Training Cycles with Global Targeting."""
        logging.info(f"Starting Training: {self.training_cycles} cycles.")

        time_step = self.train_env.current_time_step()
        policy_state = self.driver.policy.get_initial_state(self.train_env.batch_size)
        self.driver.run = common.function(self.driver.run)

        iteration = 0

        # Calculate the absolute target for the entire session to avoid drift
        total_session_target = self.training_cycles * self.episodes_per_cycle

        with self.summary_writer.as_default():
            for cycle in range(self.training_cycles):
                if self._is_interrupted:
                    logging.info("Training interrupted by user.")
                    break

                # Calculate the Absolute Target for this cycle
                # Cycle is 0-indexed, so we use (cycle + 1)
                current_cycle_target = (cycle + 1) * self.episodes_per_cycle

                # If previous cycles overshot massively, we might already be past this cycle's target.
                # In that case, we skip the Dojo grind but still perform the Showcase.
                current_episodes = self.num_episodes_metric.result().numpy()

                if current_episodes < current_cycle_target:
                    logging.info(
                        f"Cycle {cycle + 1}/{self.training_cycles}: Grinding from {current_episodes} to {current_cycle_target} episodes."
                    )
                else:
                    logging.info(
                        f"Cycle {cycle + 1}/{self.training_cycles}: Target {current_cycle_target} already met ({current_episodes}). Skipping grind."
                    )

                # DOJO PHASE: Run until we hit the GLOBAL target for this timestamp
                while self.num_episodes_metric.result().numpy() < current_cycle_target:
                    if self._is_interrupted:
                        break

                    try:
                        time_step, policy_state = self.driver.run(
                            time_step=time_step, policy_state=policy_state
                        )
                    except Exception as e:
                        logging.error(f"Critical error during collection: {e}")
                        self._is_interrupted = True
                        break

                    self._run_training_step(iteration)

                    self.replay_buffer.clear()
                    if iteration % 10 == 0:
                        gc.collect()

                    iteration += 1

                # STAGE PHASE: Showcase happens regardless of overshoot
                if not self._is_interrupted:
                    # Refresh count strictly for logging
                    final_count = self.num_episodes_metric.result().numpy()
                    self._run_showcase(final_count)

        if not self._is_interrupted:
            logging.info(
                f"Training finished. Target: {total_session_target}. "
                f"Actual: {self.num_episodes_metric.result().numpy()}."
            )
        self._graceful_shutdown()

    def _run_showcase(self, current_episodes):
        """Runs the evaluator and sends the result to the session."""
        try:
            replay_json = self.evaluator.run_showcase(self.agent.policy)

            # Send the clean showcase to the client
            self.session.replay_queue.put_nowait(replay_json)
            logging.info(f"🎥 Showcase sent at episode {current_episodes}")
        except Exception as e:
            logging.error(f"Failed to run showcase: {e}")

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

            # Use the compiled train function with reduce_retracing=True
            loss_info = self._train_fn(experience)

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
