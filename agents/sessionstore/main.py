# --- Add near the top ---
import time, threading
from typing import Dict

class SessionStore:
    def __init__(self, inactivity_seconds: int = 300):
        self._lock = threading.Lock()
        self._last_activity: Dict[str, float] = {}
        self.inactivity_seconds = inactivity_seconds
        self._session_seq: Dict[str, int] = {}  # bump to force a new thread_id

    def next_thread_id(self, session_id: str) -> str:
        with self._lock:
            now = time.time()
            last = self._last_activity.get(session_id)
            if last is None or (now - last) > self.inactivity_seconds:
                # idle too long -> new logical thread for same session
                self._session_seq[session_id] = self._session_seq.get(session_id, 0) + 1
            self._last_activity[session_id] = now
            seq = self._session_seq.get(session_id, 0)
            # derive a unique thread id that LangGraph uses for checkpointing
            return f"{session_id}::v{seq}"
        

        