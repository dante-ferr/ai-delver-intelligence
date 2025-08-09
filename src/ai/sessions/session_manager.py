import uuid
from threading import Lock
from typing import TYPE_CHECKING, Any
import asyncio
import multiprocessing as mp

if TYPE_CHECKING:
    from level import Level
    from ai.trainer.trainer import Trainer


class TrainingSession:
    """Holds all resources for a single, isolated training run."""

    def __init__(self, level: "Level"):
        self.session_id: str = str(uuid.uuid4())
        self.level: "Level" = level
        self.trainer: "Trainer | None" = None

        # This is the FAST queue for the asyncio server (WebSocket).
        self.replay_queue: asyncio.Queue[Any] = asyncio.Queue()
        # This is the PROCESS-SAFE queue for the TF-Agents environment.
        self.mp_replay_queue: mp.Queue[Any] | None = None


class SessionManager:
    """A singleton to manage all active training sessions."""

    def __init__(self):
        self._sessions: dict[str, TrainingSession] = {}
        self._lock = Lock()

    def create_session(self, level: "Level"):
        with self._lock:
            session = TrainingSession(level)
            self._sessions[session.session_id] = session
            return session

    def get_session(self, session_id):
        return self._sessions.get(session_id)

    def delete_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]


session_manager = SessionManager()
