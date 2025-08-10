import logging
import multiprocessing as std_mp
from ai.sessions.session_manager import session_manager
from ai.trainer import Trainer
from ai.sessions import REGISTRY_LOCK, SESSION_REGISTRY
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio


def run_training_in_background(session_id: str, loop: "asyncio.AbstractEventLoop"):
    """Initializes shared objects for a specific session and starts training."""
    global SESSION_REGISTRY, REGISTRY_LOCK

    session = session_manager.get_session(session_id)
    if not session:
        logging.error(f"FATAL: Background worker could not find session {session_id}.")
        return

    with REGISTRY_LOCK:
        SESSION_REGISTRY[session_id] = {
            "frame_counter": std_mp.Value("i", 0),
            "frame_lock": std_mp.Lock(),
        }

    try:
        trainer = Trainer(level=session.level, session=session, loop=loop)
        trainer.train()
    except Exception as e:
        logging.error(
            f"Error during training for session {session_id}: {e}", exc_info=True
        )
    finally:
        with REGISTRY_LOCK:
            if session_id in SESSION_REGISTRY:
                del SESSION_REGISTRY[session_id]

        logging.info(f"Cleaning up session: {session_id}")
        session_manager.delete_session(session_id)
