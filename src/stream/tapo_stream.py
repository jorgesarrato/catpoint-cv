import cv2
import logging
import os
import threading
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class TapoStream:
    """Threaded RTSP stream reader for Tapo cameras."""

    def __init__(self, url: Optional[str] = None):
        if url is None:
            user = os.getenv("TAPO_USERNAME")
            password = os.getenv("TAPO_PASSWORD")
            ip = os.getenv("TAPO_IP")
            missing = [name for name, val in [
                ("TAPO_USERNAME", user), ("TAPO_PASSWORD", password), ("TAPO_IP", ip)
            ] if not val]
            if missing:
                raise ValueError(
                    f"Missing required environment variables: {', '.join(missing)}. "
                    "Set them in .env or export them before running."
                )
            url = f"rtsp://{user}:{password}@{ip}:554/stream1"

        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.ret: bool = False
        self.frame: Optional[np.ndarray] = None
        self.stopped: bool = False
        self._lock = threading.Lock()

    def start(self) -> "TapoStream":
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self) -> None:
        backoff = 1.0
        max_backoff = 30.0

        while not self.stopped:
            if not self.cap.isOpened():
                if not self._reconnect(backoff):
                    backoff = min(backoff * 2, max_backoff)
                    continue
                backoff = 1.0

            ret, frame = self.cap.read()
            if not ret:
                if not self._reconnect(backoff):
                    backoff = min(backoff * 2, max_backoff)
                    continue
                backoff = 1.0
                continue

            with self._lock:
                self.ret = ret
                self.frame = frame

    def _reconnect(self, backoff: float) -> bool:
        """Attempt to reconnect to the RTSP stream after a failure."""
        self.cap.release()
        logger.warning("Connection lost. Retrying in %.0fs...", backoff)
        time.sleep(backoff)
        if self.stopped:
            return False
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            logger.info("Reconnected.")
            return True
        return False

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def is_alive(self) -> bool:
        return not self.stopped and self.cap.isOpened()

    def stop(self) -> None:
        self.stopped = True
        self.cap.release()

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
