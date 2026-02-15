import json
from pathlib import Path
from typing import Any, Dict
import os
from level import config as level_config

class Config:
    """
    A class to hold and provide access to configuration settings from a JSON file.
    It automatically normalizes reward values to the [-1.0, 1.0] range based on the
    largest absolute reward found, ensuring neural network stability.
    """

    def __init__(self, config_path: str = "src/ai/config.json"):
        self._config_path = Path(config_path)

        # Load the raw values (Human Readable)
        self._raw_data = self._load_config()

        # Create the operational data dict (Machine Readable)
        self._data = self._raw_data.copy()

        # Calculate and apply normalization
        self.REWARD_SCALE_FACTOR = 1.0
        self._normalize_rewards()

        # Environment Batch Configuration
        self.ENV_BATCH_SIZE = int(os.getenv("AI_BATCH_SIZE", 32))

        # Calculate steps per iteration for the Driver
        # Formula: Batch * Seconds * Actions/s
        seconds = self._data.get("collect_seconds_per_env", 10)
        aps = self._data.get("actions_per_second", 10)

        self.COLLECT_STEPS_PER_ITERATION = int(self.ENV_BATCH_SIZE * seconds * aps)

        self.TILE_WIDTH = level_config.TILE_WIDTH
        self.TILE_HEIGHT = level_config.TILE_HEIGHT
        self.DELVER_GOAL_DISTANCE_NORM = [
            level_config.MAX_GRID_SIZE[0] * self.TILE_WIDTH,
            level_config.MAX_GRID_SIZE[1] * self.TILE_HEIGHT,
        ]

    def _load_config(self) -> dict:
        try:
            with open(self._config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback for CI/CD or testing where file might not exist yet
            return {}

    def _normalize_rewards(self):
        """
        Identifies the maximum absolute reward value in the config and scales
        all reward-related keys so they fit within [-1.0, 1.0].
        """
        reward_keys = [
            "not_finished_reward",
            "finished_reward",
            "turn_reward",
            "frame_step_reward",
            "tile_exploration_reward",
            "jump_reward",
            "goal_distance_reward_scale",
            "dynamic_exam_pass_reward",
            "dynamic_target_reward",
        ]

        # 1. Find the Maximum Absolute Value (The Anchor)
        max_val = 0.0
        for key in reward_keys:
            val = self._raw_data.get(key, 0.0)
            if abs(val) > max_val:
                max_val = abs(val)

        # 2. Calculate Scale Factor
        if max_val > 0:
            self.REWARD_SCALE_FACTOR = 1.0 / max_val
        else:
            self.REWARD_SCALE_FACTOR = 1.0

        # 3. Apply Scaling to the operational data
        for key in reward_keys:
            if key in self._data:
                self._data[key] = self._data[key] * self.REWARD_SCALE_FACTOR

    def __getattr__(self, name: str) -> Any:
        # Converts Python's UPPER_SNAKE_CASE to json's snake_case for lookup
        key = name.lower()
        if key in self._data:
            return self._data[key]
        raise AttributeError(
            f"Configuration '{self._config_path}' has no setting '{key}'"
        )

    @property
    def RAW_CONFIG(self) -> Dict[str, Any]:
        """Access to the original, unscaled configuration values."""
        return self._raw_data


# A single, global instance for easy access throughout the application.
config = Config()
