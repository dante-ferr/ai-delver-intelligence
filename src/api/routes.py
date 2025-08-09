from fastapi import FastAPI, WebSocket, BackgroundTasks, APIRouter
from pydantic import BaseModel
import base64
import dill
import asyncio
from ai.sessions.session_manager import session_manager
from ai.trainer.background_trainer import run_training_in_background
import logging

router = APIRouter()


class TrainRequest(BaseModel):
    level: str  # base64 encoded dill object


@router.post("/train")
async def train_agent(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Receives a training request, creates a session, starts training in the
    background, and immediately returns the session ID.
    """
    level_data = dill.loads(base64.b64decode(request.level))

    new_session = session_manager.create_session(level_data)
    background_tasks.add_task(run_training_in_background, new_session.session_id)

    return {"message": "Training started.", "session_id": new_session.session_id}


@router.websocket("/replay/{session_id}")
async def websocket_replay_endpoint(websocket: WebSocket, session_id: str):
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
        logging.info(f"WebSocket {session_id}: Listening for replay data.")
        while True:
            json_string = await replay_queue.get()

            logging.info(f"WebSocket {session_id}: Data received. Sending to client.")
            print(json_string)
            await websocket.send_text(json_string)

    except asyncio.CancelledError:
        logging.warning(f"WebSocket for session {session_id} was cancelled.")
    except Exception as e:
        logging.error(
            f"Error in websocket for session {session_id}: {e}", exc_info=True
        )
    finally:
        logging.info(f"WebSocket for session {session_id} closed.")
