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


class TrainerController:
    def __init__(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/train/" + current_time
        self.summary_writer = create_file_writer(log_dir)

        mp.enable_interactive_mode()
        self._setup_env_and_agent()

    def _setup_env_and_agent(self):
        # Create parallel Python environments
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda i=i: LevelEnvironment(env_id=i) for i in range(ENV_BATCH_SIZE)],
            start_serially=False,
        )
        # Wrap them in a TensorFlow environment
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

    def train(self):
        print(f"Starting training for {NUM_ITERATIONS} iterations...")
        # The driver runs the whole collection phase in optimized TensorFlow graph mode
        self.driver.run = common.function(self.driver.run)

        with self.summary_writer.as_default():
            for iteration in range(NUM_ITERATIONS):
                # 1. Collect experience
                self.driver.run()

                # 2. Prepare and train
                experience = self.replay_buffer.gather_all()
                loss_info = self.agent.train(experience)
                self.replay_buffer.clear()

                # Get the current training step
                step = self.agent.train_step_counter

                if iteration % LOG_INTERVAL == 0:
                    print(
                        f"Iteration {iteration}: Step = {step}, Loss = {loss_info.loss.numpy()}"
                    )

                    # --- Log Metrics to TensorBoard ---
                    avg_return = self.avg_return_metric.result()
                    tf.summary.scalar("average_return", avg_return, step=step)
                    self.avg_return_metric.reset()  # Reset for the next interval

                    avg_length = self.avg_episode_length_metric.result()
                    tf.summary.scalar("average_episode_length", avg_length, step=step)
                    self.avg_episode_length_metric.reset()

                    tf.summary.scalar("loss", loss_info.loss, step=step)

        print("Training finished.")

    def reset(self):
        self._setup_env_and_agent()

    # @property
    # def _initial_policy(self):
    #     policy_factories = {
    #         "continuity": lambda: ContinuityRandomPolicy(
    #             self.train_env.time_step_spec(),
    #             self.train_env.action_spec(),
    #             self.train_env,
    #         )
    #     }
    #     return policy_factories[INITIAL_POLICY_NAME]()
