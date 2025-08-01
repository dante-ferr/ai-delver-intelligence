import threading
import asyncio
import websockets
import json
from queue import Queue, Empty


class SimulationSocketWorker(threading.Thread):
    def __init__(self, uri, action_queue: Queue, result_queue: Queue):
        super().__init__(daemon=True)
        self.uri = uri
        self.action_queue = action_queue
        self.result_queue = result_queue

    def run(self):
        asyncio.run(self.websocket_loop())

    async def websocket_loop(self):
        async with websockets.connect(self.uri) as websocket:
            while True:
                action = self.action_queue.get()
                await websocket.send(json.dumps(action))
                message = await websocket.recv()
                result = json.loads(message)
                self.result_queue.put(result)
