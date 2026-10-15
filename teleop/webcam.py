"""
A webcam reader that always hands you the NEWEST frame.

cv2.VideoCapture keeps a queue of old frames. If our program is slower than the camera,
cap.read() gives us frames that are already a few hundred milliseconds old and the robot
lags behind you. Reading in a background thread and keeping only the latest frame fixes that.
"""
import sys
import threading
import time

import cv2


class Webcam:
    def __init__(self, source=0, width=640, height=480, record_path=None):
        if isinstance(source, int) and sys.platform == "win32":
            # on Windows the DirectShow backend opens webcams much faster than the default one
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera/video {source!r}.\n"
                "  Mac: System Settings > Privacy & Security > Camera > allow PyCharm (or Terminal), "
                "then restart PyCharm.\n"
                "  Windows: Settings > Privacy & security > Camera > turn on 'Let desktop apps access "
                "your camera', and close other apps that use the camera (Zoom, Teams...).\n"
                "  Still stuck? Try WEBCAM_INDEX = 1 in config.py."
            )

        # a video file should play at its normal speed, a webcam just goes as fast as it can
        self.is_file = isinstance(source, str)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / fps if (self.is_file and fps > 0) else 0.0

        self.record_path = record_path   # if set, every frame is also saved to this video file
        self.recorder = None

        self.frame = None
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:   # loop the video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            if self.record_path:
                if self.recorder is None:
                    h, w = frame.shape[:2]
                    self.recorder = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                self.recorder.write(frame)
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            if self.frame_time:
                time.sleep(self.frame_time)

    def read(self):
        """Returns (frame_id, frame). frame is None until the camera has started."""
        with self.lock:
            if self.frame is None:
                return self.frame_id, None
            return self.frame_id, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        if self.recorder is not None:
            self.recorder.release()
            print("Saved the webcam video to", self.record_path)
