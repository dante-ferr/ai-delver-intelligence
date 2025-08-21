import uuid
from threading import Lock
from typing import Any
import asyncio
import multiprocessing as mp
import logging
from ai.trainer._trainer_factory import trainer_factory


class TrainingSession:
    """Holds all resources for a single, isolated training run."""

    def __init__(self, level_json: dict, amount_of_episodes: int):
        self.level_json: dict = level_json
        self.amount_of_episodes = amount_of_episodes

        self.session_id: str = str(uuid.uuid4())
        self.trainer = trainer_factory(self)

        # This is the FAST queue for the asyncio server (WebSocket).
        self.replay_queue: asyncio.Queue[Any] = asyncio.Queue()
        # This is the PROCESS-SAFE queue for the TF-Agents environment.
        self.mp_replay_queue: mp.Queue[Any] | None = None


class SessionManager:
    """A singleton to manage all active training sessions."""

    def __init__(self):
        self._sessions: dict[str, TrainingSession] = {}
        self._lock = Lock()

    def create_session(self, level_json: dict, amount_of_episodes: int):
        with self._lock:
            session = TrainingSession(level_json, amount_of_episodes)
            self._sessions[session.session_id] = session
            return session

    def get_session(self, session_id):
        return self._sessions.get(session_id)

    def delete_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                try:
                    session.replay_queue.put_nowait("end")
                except asyncio.QueueFull:
                    logging.warning(
                        f"Could not signal end to session {session_id}: queue is full."
                    )

                del self._sessions[session_id]


session_manager = SessionManager()
