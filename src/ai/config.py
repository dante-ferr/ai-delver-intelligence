import json
from pathlib import Path
from typing import Any


class Config:
    """A class to hold and provide access to configuration settings from a JSON file."""

    def __init__(self, config_path: str = "src/ai/config.json"):
        self._config_path = Path(config_path)
        self._data = self._load_config()

    def _load_config(self) -> dict:
        with open(self._config_path, "r") as f:
            return json.load(f)

    def __getattr__(self, name: str) -> Any:
        # Converts Python's UPPER_SNAKE_CASE to json's snake_case for lookup
        key = name.lower()
        if key in self._data:
            return self._data[key]
        raise AttributeError(
            f"Configuration '{self._config_path}' has no setting '{key}'"
        )


# A single, global instance for easy access throughout the application.
config = Config()
