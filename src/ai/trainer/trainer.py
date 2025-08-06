import tensorflow as tf
from tf_agents.environments import TFPyEnvironment, parallel_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import multiprocessing as mp
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from ai.environments.level.environment import LevelEnvironment
import datetime
from ai.agents import PPOAgentFactory
from tensorflow.summary import create_file_writer  # type: ignore
from ai.config import (
    LEARNING_RATE,
    GAMMA,
    NUM_ITERATIONS,
    REPLAY_BUFFER_CAPACITY,
    LOG_INTERVAL,
    ENV_BATCH_SIZE,
    COLLECT_STEPS_PER_ITERATION,
)
from multiprocessing import Process, Manager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.connection_manager import ConnectionManager


def _create_level_environment(env_id: int, replay_queue=None):
    """Factory function for creating an environment."""
    return LevelEnvironment(env_id=env_id, replay_queue=replay_queue)


def replay_queue_worker(replay_queue, connection_manager):
    """
    A worker function that runs in a separate process.
    It continuously checks the queue and sends replays.
    """
    import asyncio

    async def send_data(data):
        if connection_manager:
            await connection_manager.send_replay_data(data)

    print("▶️ Replay worker process started.")
    while True:
        if not replay_queue.empty():
            replay_data = replay_queue.get()
            asyncio.run(send_data(replay_data))


class Trainer:
    def __init__(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/train/" + current_time
        self.summary_writer = create_file_writer(log_dir)

        self.manager = Manager()
        self.replay_queue = self.manager.Queue()

        mp.enable_interactive_mode()
        self._setup_env_and_agent()

    def _setup_env_and_agent(self):
        # Parallel environments creation
        replay_queue = self.replay_queue
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [
                lambda i=i: _create_level_environment(
                    env_id=i, replay_queue=replay_queue
                )
                for i in range(ENV_BATCH_SIZE)
            ],
            start_serially=False,
        )

        self.train_env = TFPyEnvironment(py_env)

        self.agent = PPOAgentFactory(
            self.train_env, learning_rate=LEARNING_RATE, gamma=GAMMA
        ).get_agent()
        self.agent.train_step_counter.assign(0)

        # Create the replay buffer suitable for PPO (on-policy)
        # It will store trajectories collected by the agent
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=ENV_BATCH_SIZE,  # Batched envs
            max_length=REPLAY_BUFFER_CAPACITY,
        )

        # Create metrics to track the return (total reward) and episode length
        self.avg_return_metric = tf_metrics.AverageReturnMetric(
            batch_size=ENV_BATCH_SIZE
        )
        self.avg_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=ENV_BATCH_SIZE
        )

        self.driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.train_env,
            self.agent.collect_policy,
            # Add the metrics to the driver's observers list
            observers=[
                self.replay_buffer.add_batch,
                self.avg_return_metric,
                self.avg_episode_length_metric,
            ],
            num_episodes=COLLECT_STEPS_PER_ITERATION,
        )

    def train(self, connection_manager: "None | ConnectionManager" = None):
        print(f"Starting training for {NUM_ITERATIONS} iterations...")
        self.driver.run = common.function(self.driver.run)

        # Start the replay worker in a separate, daemonic process
        replay_process = Process(
            target=replay_queue_worker,
            args=(self.replay_queue, connection_manager),
            daemon=True,  # This ensures the process exits when the main script does
        )
        replay_process.start()

        with self.summary_writer.as_default():
            for iteration in range(NUM_ITERATIONS):
                self.driver.run()
                experience = self.replay_buffer.gather_all()
                loss_info = self.agent.train(experience)
                self.replay_buffer.clear()

                step = self.agent.train_step_counter
                if iteration % LOG_INTERVAL == 0:
                    print(
                        f"Iteration {iteration}: Step = {step}, Loss = {loss_info.loss.numpy()}"
                    )
                    self._handle_tensorboard(loss_info, step)

        print("Training finished.")

    def _handle_tensorboard(self, loss_info, step):
        avg_return = self.avg_return_metric.result()
        tf.summary.scalar("average_return", avg_return, step=step)
        self.avg_return_metric.reset()  # Reset for the next interval

        avg_length = self.avg_episode_length_metric.result()
        tf.summary.scalar("average_episode_length", avg_length, step=step)
        self.avg_episode_length_metric.reset()

        tf.summary.scalar("loss", loss_info.loss, step=step)

    def reset(self):
        self._setup_env_and_agent()
