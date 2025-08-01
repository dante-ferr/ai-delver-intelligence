import logging
import datetime
import time


class LevelEnvironmentLogger:
    def __init__(self, env_id, log_dir="./logs"):
        self.logger = logging.getLogger(f"ai_delver_environment_{env_id}")
        self.logger.setLevel(logging.DEBUG)

        log_file_name = (
            f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_env_{env_id}.log'
        )
        file_handler = logging.FileHandler(f"{log_dir}/{log_file_name}")
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(episode)s - %(step)s - %(reward)s - %(delver_position)s - %(goal_position)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.last_step_log_time = time.time()

    def log_episode_start(self, episode):
        self.logger.info("Episode start", extra={"episode": episode})

    def handle_step(
        self, reward, move, move_angle, delver_position, global_frame_count, fps
    ):
        current_time = time.time()

        if (current_time - self.last_step_log_time) >= 0.5:
            self.logger.info(
                "Step",
                extra={
                    "reward": reward,
                    "move": move,
                    "move_angle": move_angle,
                    "delver_position": delver_position,
                    "step": global_frame_count,
                    "fps": fps,
                },
            )

            self.last_step_log_time = current_time

    def log_episode_end(self, episode, reward):
        self.logger.info("Episode end", extra={"episode": episode, "reward": reward})

    def log_error(self, error_message):
        self.logger.error("Error", extra={"error": error_message})
