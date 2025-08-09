from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing import Manager

class TrainerReplayManager:
    """Manages the replay system for the trainer."""

    def __init__(self, manager):
        self.manager = manager
        self.replay_queue = self.manager.Queue()
