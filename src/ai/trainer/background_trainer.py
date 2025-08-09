import logging
import asyncio
import threading
import multiprocessing as std_mp
from queue import Empty
from ai.sessions.session_manager import session_manager
from ai.trainer.trainer import Trainer


# The queue_translator worker function is correct and does not need changes.
def queue_translator(
    loop: asyncio.AbstractEventLoop,
    mp_queue: std_mp.Queue,
    async_queue: asyncio.Queue,
    stop_event: threading.Event,
):
    while not stop_event.is_set():
        try:
            data = mp_queue.get(timeout=1.0)
            asyncio.run_coroutine_threadsafe(async_queue.put(data), loop)
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Error in queue_translator: {e}", exc_info=True)


def run_training_in_background(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        logging.error(f"FATAL: Background worker could not find session {session_id}.")
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # ✅ Use the simpler, lightweight multiprocessing objects.
    # This avoids the complex Manager process and prevents the conflict.
    mp_queue = std_mp.Queue()
    frame_counter = std_mp.Value("i", 0)
    frame_lock = std_mp.Lock()

    # Store the multiprocessing queue on the session for the translator to use.
    session.mp_replay_queue = mp_queue
    async_queue = session.replay_queue

    stop_event = threading.Event()
    translator = threading.Thread(
        target=queue_translator,
        args=(loop, mp_queue, async_queue, stop_event),
        daemon=True,
    )
    translator.start()

    try:
        # Create the Trainer, passing it the new, simpler objects.
        trainer = Trainer(
            level=session.level,
            replay_queue=mp_queue,
            frame_counter=frame_counter,
            frame_lock=frame_lock,
        )
        trainer.train()
    finally:
        logging.info(f"Stopping translator for session {session_id}")
        stop_event.set()
        translator.join(timeout=2.0)
        session_manager.delete_session(session_id)
