from tf_agents.environments import TFPyEnvironment, parallel_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import multiprocessing as mp
from tf_agents.utils import common
from ai.environments.level.environment import LevelEnvironment
from ai.agents import PPOAgentFactory
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

        # Create a driver to run the collection loop efficiently
        self.driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.train_env,
            self.agent.collect_policy,  # Use the agent's policy for collection
            observers=[self.replay_buffer.add_batch],
            num_episodes=COLLECT_STEPS_PER_ITERATION,
        )

    def train(self):
        print(f"Starting training for {NUM_ITERATIONS} iterations...")
        # The driver runs the whole collection phase in optimized TensorFlow graph mode
        self.driver.run = common.function(self.driver.run)

        for iteration in range(NUM_ITERATIONS):
            # 1. Collect a few episodes of experience using the current policy
            self.driver.run()

            # 2. Prepare the collected data for training
            experience = self.replay_buffer.gather_all()

            # 3. Train the agent on this data for a few epochs
            loss_info = self.agent.train(experience)

            # 4. Clear the buffer for the next collection phase
            self.replay_buffer.clear()

            if iteration % LOG_INTERVAL == 0:
                print(f"Iteration {iteration}: Loss = {loss_info.loss.numpy()}")

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
