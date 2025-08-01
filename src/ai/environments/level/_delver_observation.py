from typing import TypedDict
import numpy as np


class DelverObservation(TypedDict):
    walls: np.ndarray
    delver_position: np.ndarray
    goal_position: np.ndarray
