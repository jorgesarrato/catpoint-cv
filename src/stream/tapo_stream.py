import cv2
import os
import threading
import time
from typing import Optional
import numpy as np


class TapoStream:
    """Threaded RTSP stream reader for Tapo cameras."""

    def __init__(self, url: Optional[str] = None):
        if url is None:
            user = os.getenv("TAPO_USERNAME")
            password = os.getenv("TAPO_PASSWORD")
            ip = os.getenv("TAPO_IP")
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
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def is_alive(self) -> bool:
        return not self.stopped and self.cap.isOpened()

    def stop(self) -> None:
        self.stopped = True
        self.cap.release()
