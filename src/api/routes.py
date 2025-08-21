from fastapi import FastAPI, WebSocket, BackgroundTasks, APIRouter
from pydantic import BaseModel
import asyncio
from ai.sessions.session_manager import session_manager
from ai.trainer.background_trainer import run_training_in_background
import logging
from ai.utils.log_bytes_size import log_bytes_size

router = APIRouter()


class TrainRequest(BaseModel):
    level: dict  # base64 encoded dill object
    amount_of_episodes: int


@router.post("/train")
async def train_agent(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Receives a training request, creates a session, starts training in the
    background, and immediately returns the session ID.
    """
    new_session = session_manager.create_session(
        request.level, request.amount_of_episodes
    )

    background_tasks.add_task(run_training_in_background, new_session.session_id)

    return {"message": "Training started.", "session_id": new_session.session_id}


@router.post("/interrupt-training/{session_id}")
async def interrupt_training(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        return {"message": "Session not found.", "success": False}
    if not session.trainer:
        return {
            "message": f"Training not started for session with id {session_id}.",
            "success": False,
        }

    session.trainer.interrupt_training()
    return {"message": "Training interrupted.", "success": True}


@router.websocket("/episode-trajectory/{session_id}")
async def websocket_training_endpoint(websocket: WebSocket, session_id: str):
    """
    Handles the WebSocket connection, consuming directly from the
    native asyncio queue.
    """
    await websocket.accept()
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Get the asyncio queue directly from the session.
    # The background translator is responsible for filling it.
    replay_queue = session.replay_queue

    try:
        while session_manager.get_session(session_id):
            replay_string = await replay_queue.get()

            if replay_string == "end":
                logging.info(
                    f"WebSocket for session {session_id} is preparing to close."
                )
                await websocket.send_json({"end": True})
                await websocket.close()
                break

            await websocket.send_json({"trajectory": replay_string})

    except asyncio.CancelledError:
        logging.warning(f"WebSocket for session {session_id} was cancelled.")
    except Exception as e:
        logging.error(
            f"Error in websocket for session {session_id}: {e}", exc_info=True
        )
    finally:
        logging.info(f"WebSocket for session {session_id} closed.")
