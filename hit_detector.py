import threading
import queue
import time

try:
    import mss
    import cv2
    import numpy as np
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


if AVAILABLE:
    class HitDetector:
        """
        Detects Marvel Rivals hit markers by watching a small region around the
        screen centre at ~60 fps.

        Strategy: compare consecutive frames and count pixels that suddenly became
        significantly brighter. Camera motion changes pixel colours but rarely
        causes a concentrated brightness spike; a white hit marker does.

        The hit marker appears *after* the shot registers on the server, so each
        detected timestamp needs a 50–300 ms lookback in the analyser to find
        the stick state at the moment the shot was actually fired.
        """

        _REGION_PX    = 60     # side length of the capture square in pixels
        _POLL_SEC     = 1/60   # target capture interval
        _BRIGHT_GAIN  = 40     # brightness units a pixel must gain to count
        _MIN_PIXELS   = 12     # how many such pixels trigger a hit
        _COOLDOWN     = 0.20   # minimum seconds between consecutive hits (debounce)

        def __init__(self, screen_cx: int, screen_cy: int):
            half = self._REGION_PX // 2
            self._region = {
                'left':   screen_cx - half,
                'top':    screen_cy - half,
                'width':  self._REGION_PX,
                'height': self._REGION_PX,
            }
            self._hits: queue.Queue = queue.Queue()
            self._running = False
            self._last_hit = 0.0

        def start(self) -> None:
            self._running = True
            threading.Thread(target=self._loop, daemon=True).start()

        def stop(self) -> None:
            self._running = False

        def get_hits(self) -> list:
            """Drain and return all pending hit timestamps."""
            hits = []
            while True:
                try:
                    hits.append(self._hits.get_nowait())
                except queue.Empty:
                    break
            return hits

        def _loop(self) -> None:
            with mss.mss() as sct:
                prev = None
                while self._running:
                    t0 = time.perf_counter()

                    frame = np.asarray(sct.grab(self._region), dtype=np.uint8)
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY).astype(np.int16)

                    if prev is not None:
                        # Only count pixels that became brighter — not general churn
                        gained = np.clip(gray - prev, 0, 255)
                        if int(np.sum(gained > self._BRIGHT_GAIN)) >= self._MIN_PIXELS:
                            now = time.time()
                            if now - self._last_hit >= self._COOLDOWN:
                                self._hits.put(now)
                                self._last_hit = now

                    prev = gray
                    elapsed = time.perf_counter() - t0
                    wait = self._POLL_SEC - elapsed
                    if wait > 0:
                        time.sleep(wait)

else:
    class HitDetector:
        """Stub used when mss / opencv-python are not installed."""
        def __init__(self, *_, **__): pass
        def start(self): pass
        def stop(self): pass
        def get_hits(self): return []
