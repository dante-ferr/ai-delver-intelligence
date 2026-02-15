import asyncio
import tensorflow as tf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio


class WebSocketObserver:
    def __init__(self, async_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._async_queue = async_queue
        self._loop = loop

    def _send_to_queue(self, json_tensor: tf.Tensor):
        for json_string_bytes in json_tensor.numpy():  # type: ignore
            if json_string_bytes:
                json_string = json_string_bytes.decode("utf-8")
                self._loop.call_soon_threadsafe(
                    self._async_queue.put_nowait, json_string
                )

    def __call__(self, trajectory):
        """
        This method is called by the driver inside the TensorFlow graph.
        It now correctly returns the TensorFlow operation.
        """
        replay_json_tensor = trajectory.observation["replay_json"]

        # Return the operation created by tf.py_function so it is executed by the driver.
        return tf.py_function(
            func=self._send_to_queue,
            inp=[replay_json_tensor],
            Tout=[],
        )
