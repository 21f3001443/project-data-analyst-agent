import time, threading, uuid
from typing import Dict

class SessionStore:
    def __init__(self, inactivity_seconds: int = 300):
        self._lock = threading.Lock()
        self._last_activity: Dict[str, float] = {}
        self.inactivity_seconds = inactivity_seconds
        self._session_seq: Dict[str, int] = {}

    def next_thread_id(self, session_id: str, always_new: bool = False) -> str:
        with self._lock:
            now = time.time()
            last = self._last_activity.get(session_id)

            if always_new:
                # bump every call
                self._session_seq[session_id] = self._session_seq.get(session_id, 0) + 1
            else:
                # bump only after inactivity
                if last is None or (now - last) > self.inactivity_seconds:
                    self._session_seq[session_id] = self._session_seq.get(session_id, 0) + 1

            self._last_activity[session_id] = now
            seq = self._session_seq.get(session_id, 0)

            # Option A: sequential IDs
            # return f"{session_id}::v{seq}"

            # Option B (if you want *completely unique* IDs):
            return f"{session_id}::{uuid.uuid4().hex}"
        
    def clear(self, session_id: str = None):
        with self._lock:
            if session_id:
                self._last_activity.pop(session_id, None)
                self._session_seq.pop(session_id, None)
            else:
                self._last_activity.clear()
                self._session_seq.clear()