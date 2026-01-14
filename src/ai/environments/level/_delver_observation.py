from typing import TypedDict
import numpy as np


class DelverObservation(TypedDict):
    local_view: np.ndarray
    global_state: np.ndarray
    replay_json: np.ndarray
