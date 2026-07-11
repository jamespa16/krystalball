"""Background-thread webcam capture with a lock-protected latest-frame buffer.

Runs cap.read() in its own thread so the training/inference loop never
blocks waiting on camera I/O -- it just grabs whatever the latest frame is.
"""

import threading

import cv2


class WebcamStream:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self._lock = threading.Lock()
        self._frame = None
        self._stopped = False

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """Returns a copy of the latest frame, or None if none has arrived yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()
