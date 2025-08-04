import json

with open("src/ai/config.json", "r") as f:
    config = json.load(f)

NUM_ITERATIONS = config["num_iterations"]
INITIAL_COLLECT_STEPS = config["initial_collect_steps"]
COLLECT_STEPS_PER_ITERATION = config["collect_steps_per_iteration"]
REPLAY_BUFFER_BATCH_SIZE = config["replay_buffer_batch_size"]
REPLAY_BUFFER_CAPACITY = config["replay_buffer_capacity"]
LEARNING_RATE = config["learning_rate"]
GAMMA = config["gamma"]
EPSILON_GREEDY = config["epsilon_greedy"]
LOG_INTERVAL = config["log_interval"]
CHECKPOINT_INTERVAL = config["checkpoint_interval"]
SEED = config["seed"]
ENTROPY_REGULARIZATION = config["entropy_regularization"]
INITIAL_POLICY = config["initial_policy"]
ENV_BATCH_SIZE = config["env_batch_size"]
MAX_FRAMES_PER_EPISODE = config["max_frames_per_episode"]
FPS = config["fps"]
