# from tf_agents.environments import tf_py_environment
# from .agents import PPOAgentFactory
# from .environments import LevelEnvironment


# def main():
#     # Create environment and agent
#     test_env = tf_py_environment.TFPyEnvironment(LevelEnvironment())
#     agent = PPOAgentFactory(test_env).get_agent()

#     print("\nTesting trained agent...")
#     policy = agent.policy

#     for _ in range(10):
#         time_step = test_env.reset()
#         target = test_env.pyenv.envs[0]._target
#         print(f"\nTarget number: {target}")

#         for t in range(3):  # Give it 3 tries
#             action_step = policy.action(time_step)
#             guess = action_step.action.numpy()[0]
#             print(f"Guess {t+1}: {guess}")

#             time_step = test_env.step(action_step.action)

#             if guess == target:
#                 print("Correct!")
#                 break


# if __name__ == "__main__":
#     main()
