import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import greedy_policy
from tf_agents.utils import common
from ai.environments.level.environment import LevelEnvironment
from ai.config import config
import logging
from typing import Any

class Evaluator:
    """
    Orchestrates the 'Stage' (Showcase) phase and Competence Exams.
    Uses a separate environment instance to run deterministic episodes.
    """

    def __init__(self, level_json: dict, session_id: str):
        self.level_json = level_json
        self.session_id = session_id

        # env_id 999 prevents this environment from logging to the main training logs
        self._py_env = LevelEnvironment(
            env_id=999, level_json=level_json, session_id=session_id, is_showcase=True
        )
        self.env = tf_py_environment.TFPyEnvironment(self._py_env)

    def run_showcase(self, agent_policy) -> str:
        """
        Runs a single episode using a Greedy (Argmax) policy.
        Returns the trajectory JSON for frontend replay.
        """
        logging.info("🎬 Running Showcase Episode...")

        showcase_policy = greedy_policy.GreedyPolicy(agent_policy)
        action_fn: Any = common.function(showcase_policy.action)

        time_step: Any = self.env.reset()
        policy_state = showcase_policy.get_initial_state(self.env.batch_size)

        while not time_step.is_last():
            action_step = action_fn(time_step, policy_state)
            policy_state = action_step.state
            time_step = self.env.step(action_step.action)

        final_obs = time_step.observation
        replay_json_bytes = final_obs["replay_json"].numpy()[0]
        replay_json = replay_json_bytes.decode("utf-8")

        logging.info("✅ Showcase Episode Complete.")
        return replay_json

    def evaluate_competence(self, agent_policy, num_episodes: int) -> float:
        """
        Runs a batch of episodes using the Greedy Policy to test actual skill.
        Returns the success rate (0.0 to 1.0).
        """
        logging.info(f"🧐 Running Competence Exam ({num_episodes} episodes)...")

        showcase_policy = greedy_policy.GreedyPolicy(agent_policy)
        action_fn: Any = common.function(showcase_policy.action)

        success_count = 0

        for _ in range(num_episodes):
            time_step: Any = self.env.reset()
            policy_state = showcase_policy.get_initial_state(self.env.batch_size)

            while not time_step.is_last():
                action_step = action_fn(time_step, policy_state)
                policy_state = action_step.state
                time_step = self.env.step(action_step.action)

            # Check if the final reward meets the success target
            # reward is a tensor, we need to extract the value
            reward = time_step.reward.numpy()[0]

            if reward >= config.DYNAMIC_TARGET_REWARD:
                success_count += 1

        return success_count / num_episodes
