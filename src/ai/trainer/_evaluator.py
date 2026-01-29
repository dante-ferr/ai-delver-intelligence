import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import greedy_policy
from tf_agents.utils import common
from ai.environments.level.environment import LevelEnvironment
import logging


class Evaluator:
    """
    Orchestrates the 'Stage' (Showcase) phase.
    Runs a single, deterministic episode to demonstrate the agent's actual skill level
    without training noise.
    """

    def __init__(self, level_json: dict, session_id: str):
        self.level_json = level_json
        self.session_id = session_id

        # Instantiate a dedicated environment for showcase
        # is_showcase=True ensures we record the full trajectory
        self._py_env = LevelEnvironment(
            env_id=999, level_json=level_json, session_id=session_id, is_showcase=True
        )
        self.env = tf_py_environment.TFPyEnvironment(self._py_env)

    def run_showcase(self, agent_policy) -> str:
        """
        Runs a single episode using a Greedy (Argmax) policy.
        Returns the trajectory JSON.
        """
        logging.info("🎬 Running Showcase Episode...")

        # Wrap the stochastic training policy with a Greedy policy
        showcase_policy = greedy_policy.GreedyPolicy(agent_policy)

        # OPTIMIZATION: Wrap the action method with common.function
        # This compiles the graph once and prevents the "retracing" warning
        # inside the loop, drastically speeding up the showcase.
        action_fn = common.function(showcase_policy.action)

        time_step = self.env.reset()

        # Initialize LSTM state
        policy_state = showcase_policy.get_initial_state(self.env.batch_size)

        while not time_step.is_last():
            # Use the compiled function instead of calling .action directly
            action_step = action_fn(time_step, policy_state)

            # Update memory state
            policy_state = action_step.state

            time_step = self.env.step(action_step.action)

        # Extract trajectory JSON
        final_obs = time_step.observation
        replay_json_bytes = final_obs["replay_json"].numpy()[0]
        replay_json = replay_json_bytes.decode("utf-8")

        logging.info("✅ Showcase Episode Complete.")
        return replay_json
