import logging
import datetime
import time
import os

class ContextFilter(logging.Filter):
    """
    Ensures a default 'episode' attribute exists on all log records.
    """
    def filter(self, record):
        if not hasattr(record, "episode"):
            record.episode = "---"
        return True


class CustomFormatter(logging.Formatter):
    """
    A custom formatter that applies different formats based on the log record's content.
    """

    # Simple format for general messages (start, end, errors)
    simple_fmt = "%(asctime)s - %(message)s"

    # Detailed format specifically for step-by-step training data
    # Updated to show Jump status instead of Angle
    detailed_fmt = "%(asctime)s - Global Frame: %(global_frame_count)s | Sim Frame: %(simulation_frame)s | Reward: %(reward)s | Pos: %(delver_position)s | Run: %(run)s | Jump: %(jump)s | FPS: %(fps)s"

    def __init__(self):
        super().__init__(fmt="%(levelname)s: %(message)s", datefmt=None, style="%")
        self.simple_formatter = logging.Formatter(self.simple_fmt)
        self.detailed_formatter = logging.Formatter(self.detailed_fmt)

    def format(self, record):
        # Check if the log record is a detailed step log.
        # We use the presence of the 'simulation_frame' attribute as the trigger.
        if hasattr(record, "simulation_frame"):
            return self.detailed_formatter.format(record)

        # Otherwise, use the simple format.
        return self.simple_formatter.format(record)


class LevelEnvironmentLogger:

    def __init__(self, log_dir="./logger_train/logs"):
        self.logger = logging.getLogger(f"ai_delver_environment")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.logger.addFilter(ContextFilter())
        os.makedirs(log_dir, exist_ok=True)

        log_file_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_env.log'
        file_handler = logging.FileHandler(f"{log_dir}/{log_file_name}")
        console_handler = logging.StreamHandler()

        # Use our new CustomFormatter for both handlers
        formatter = CustomFormatter()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.last_step_log_time = time.time()

    def log_episode_start(self):
        # This call is simple. The formatter will use the simple format.
        self.logger.info("Episode Start")

    def log_step(
        self,
        reward,
        run,
        jump,
        delver_position,
        global_frame_count,
        simulation_frame,
        fps,
    ):
        current_time = time.time()
        if (current_time - self.last_step_log_time) >= 1.0:  # Throttled to 1 second
            # This call is detailed. The formatter will see the attributes in extra
            # and use the detailed format automatically.
            self.logger.info(
                "",  # The message itself is not needed as the formatter handles everything
                extra={
                    "global_frame_count": global_frame_count,
                    "simulation_frame": simulation_frame,
                    "reward": f"{reward:.4f}",
                    "delver_position": f"({delver_position[0]:.1f}, {delver_position[1]:.1f})",
                    "run": run,
                    "jump": jump,
                    "fps": f"{fps:.1f}",
                },
            )
            self.last_step_log_time = current_time

    def log_episode_end(self, reward):
        # This call is simple. The formatter will use the simple format.
        message = f"Episode End. Final Reward: {reward:.4f}"
        self.logger.info(message)

    def log_error(self, error_message):
        # This call is simple. The formatter will use the simple format.
        self.logger.error(error_message)
