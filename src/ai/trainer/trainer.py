import asyncio
import logging
import datetime
import gc
import tensorflow as tf
from tensorflow.summary import create_file_writer
from tf_agents.trajectories import time_step as ts
from ai.config import config
from ai.sessions.session_manager import session_manager
from ._evaluator import Evaluator
from ._trainer_agent_manager import TrainerAgentManager
from ._trainer_environment_manager import TrainerEnvironmentManager
from ._trainer_model_manager import TrainerModelManager
from ._training_metrics import TrainingMetrics
from typing import TYPE_CHECKING, Optional, Any, Tuple

if TYPE_CHECKING:
    from ai.sessions.session_manager import TrainingSession

BENCHMARK_MODE = False


class Trainer:
    """
    Orchestrates the training lifecycle.
    Delegates specific tasks to AgentManager, EnvManager, and Metrics.
    """

    def __init__(
        self,
        session: "TrainingSession",
        loop: "asyncio.AbstractEventLoop",
        model_bytes: None | bytes = None,
    ):
        self.session = session
        self.loop = loop
        self.model_bytes = model_bytes
        self._is_interrupted = False

        self.training_cycles = session.amount_of_cycles or 0
        self.episodes_per_cycle = session.episodes_per_cycle
        self.levels_trained = 0

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = create_file_writer(f"logs/train/{current_time}")

        self.model_manager = TrainerModelManager(self)
        self.env_manager = TrainerEnvironmentManager(session.session_id)
        self.agent_manager = TrainerAgentManager(config.LEARNING_RATE, config.GAMMA)

        self._evaluator: Optional[Evaluator] = None

    def setup_env_and_agent(self) -> None:
        """Bootstrap the first level."""
        first_level = self.session.level_jsons[0]

        self.env_manager.setup_environment(first_level)
        self._evaluator = Evaluator(first_level, self.session.session_id)

        # Uses the normalized target from config
        self._metrics = TrainingMetrics(self.env_manager.tf_env.batch_size)

        self.agent_manager.setup(
            self.env_manager.tf_env,
            self.metrics.observers,
            benchmark_mode=BENCHMARK_MODE,
        )

        if self.model_bytes:
            try:
                self.model_manager.load_serialized_model(self.model_bytes)
            except Exception as e:
                logging.error(f"Failed to load model: {e}")

    def train(self) -> None:
        """Main Loop."""
        total_levels = len(self.session.level_jsons)

        with self.summary_writer.as_default():
            for level_idx, level_json in enumerate(self.session.level_jsons):
                self.levels_trained = level_idx
                if self._is_interrupted:
                    break

                if level_idx > 0:
                    self._transition_to_level(level_json)

                self._process_level_loop(level_idx, total_levels)

        self._graceful_shutdown()

    def _process_level_loop(self, level_idx: int, total_levels: int) -> None:
        """Runs the Dojo/Stage cycles for a specific level."""
        logging.info(f"=== Starting Level {level_idx + 1}/{total_levels} ===")

        self.metrics.reset()
        tf_env = self.env_manager.tf_env
        time_step = tf_env.current_time_step()
        policy_state = self.agent_manager.agent.collect_policy.get_initial_state(
            tf_env.batch_size
        )

        cycle = 0

        while True:
            if self._is_interrupted:
                break

            if self._should_stop_loop(cycle, level_idx):
                break

            time_step, policy_state = self._run_cycle(
                cycle, level_idx, time_step, policy_state
            )

            if self._check_graduation(level_idx):
                break

            cycle += 1

    def _should_stop_loop(self, cycle: int, level_idx: int) -> bool:
        if (
            self.session.level_transitioning_mode != "dynamic"
            and cycle >= self.training_cycles
        ):
            logging.info(f"Level {level_idx+1} finished (Cycle limit).")
            return True
        return False

    def _run_cycle(
        self, cycle: int, level_idx: int, time_step, policy_state
    ) -> Tuple[Any, Any]:
        """Runs one cycle of grinding (Dojo) and one Showcase (Stage)."""
        target_episodes = (cycle + 1) * self.episodes_per_cycle

        while self.metrics.num_episodes < target_episodes:
            if self._is_interrupted:
                break

            try:
                collection_result = self.agent_manager.run_collection_step(
                    time_step, policy_state
                )

                if isinstance(collection_result, tuple) and len(collection_result) == 2:
                    time_step, policy_state = collection_result
                else:
                    raise ValueError(
                        f"run_collection_step returned an invalid result: {collection_result}"
                    )

            except Exception as e:
                logging.error(f"Collection Error: {e}")
                self._is_interrupted = True
                break

            loss_info = self.agent_manager.run_training_step()

            self.agent_manager.clear_buffer()
            step = self.agent_manager.get_step_count()

            if step % 10 == 0:
                gc.collect()
            if step % config.LOG_INTERVAL == 0:
                self._log_metrics(loss_info, step)

        if not self._is_interrupted:
            self._run_showcase(self.metrics.num_episodes)

        return time_step, policy_state

    def _check_graduation(self, level_idx: int) -> bool:
        return False

    def _transition_to_level(self, level_json: dict) -> None:
        """Re-initializes environment for the new level."""
        logging.info("Transitionsing levels...")
        self.env_manager.setup_environment(level_json)
        self._evaluator = Evaluator(level_json, self.session.session_id)

        self._metrics = TrainingMetrics(self.env_manager.tf_env.batch_size)

        self.agent_manager.setup(
            self.env_manager.tf_env,
            self.metrics.observers,
            benchmark_mode=BENCHMARK_MODE,
        )
        self.agent_manager.clear_buffer()
        gc.collect()

        # Send level transition message separately
        self.session.replay_queue.put_nowait(
            {"type": "level_transition", "levels_trained": self.levels_trained}
        )

    def _run_showcase(self, episode_count: int) -> None:
        try:
            replay_json = self.evaluator.run_showcase(self.agent_manager.get_policy())
            self.session.replay_queue.put_nowait(
                {
                    "type": "showcase",
                    "trajectory": replay_json,
                    "level_episode_count": str(episode_count),
                }
            )
        except Exception as e:
            logging.error(f"Showcase failed: {e}")

    def _log_metrics(self, loss_info, step) -> None:
        avg_ret = self.metrics.avg_return
        logging.info(
            f"Step {step}: Loss={loss_info.loss.numpy():.4f}, Return={avg_ret:.2f}"
        )

        tf.summary.scalar("average_return", avg_ret, step=step)
        tf.summary.scalar("loss", loss_info.loss, step=step)
        self.metrics.reset_avg_return()

    def _graceful_shutdown(self) -> None:
        logging.info("Shutting down trainer...")
        self.env_manager.close()
        self.agent_manager.clear_buffer()
        session_manager.delete_session(self.session.session_id)

    def interrupt_training(self) -> None:
        self._is_interrupted = True

    @property
    def metrics(self) -> TrainingMetrics:
        if not self._metrics:
            raise Exception("Metrics not initialized.")
        return self._metrics

    @property
    def evaluator(self) -> Evaluator:
        if not self._evaluator:
            raise Exception("Evaluator not initialized.")
        return self._evaluator
