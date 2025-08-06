from fastapi import WebSocket


class ConnectionManager:
    """Manages the active client WebSocket connection."""

    def __init__(self):
        self.active_connection: WebSocket | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        print("Client connected to WebSocket.")

    def disconnect(self):
        self.active_connection = None
        print("Client disconnected.")

    async def send_replay_data(self, data: dict):
        print(data)
        if self.active_connection:
            await self.active_connection.send_json(data)
            print("Replay data sent to client.")
        else:
            print("No active client connection to send replay data.")


# Create a single instance to be used across the application
manager = ConnectionManager()
