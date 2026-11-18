from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from typing import Any


# Shared state file lives at project root
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".live_state.json",
)


class LiveStateStore:
    """
    File-backed state store for current live vehicle states and metadata.
    Writes are persisted to a JSON file so that separate processes
    (e.g. realtime_monitor and Streamlit app) can share state.
    """

    _DEFAULT_META: dict[str, Any] = {
        "last_poll_time_utc": None,
        "last_feed_entity_count": 0,
        "last_vehicle_count": 0,
        "last_success": False,
        "last_error": None,
    }

    def __init__(self, state_file: str = _STATE_FILE) -> None:
        self._lock = threading.Lock()
        self._state_file = state_file

    # ---- internal persistence helpers ----

    def _read(self) -> dict[str, Any]:
        """Read the shared state from disk."""
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"vehicles": {}, "meta": dict(self._DEFAULT_META)}

    def _write(self, data: dict[str, Any]) -> None:
        """Atomically write state to disk using a temp file + rename."""
        dir_name = os.path.dirname(self._state_file)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            # On Windows, os.replace is atomic and overwrites the target.
            os.replace(tmp_path, self._state_file)
        except Exception:
            # Clean up temp file on failure
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    # ---- public API (unchanged) ----

    def update_vehicle(self, vehicle_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data["vehicles"][vehicle_id] = payload
            self._write(data)

    def bulk_replace(self, vehicles: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            data = self._read()
            data["vehicles"] = vehicles
            self._write(data)

    def get_all_vehicles(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            data = self._read()
            return deepcopy(data.get("vehicles", {}))

    def get_vehicle_count(self) -> int:
        with self._lock:
            data = self._read()
            return len(data.get("vehicles", {}))

    def update_meta(self, **kwargs: Any) -> None:
        with self._lock:
            data = self._read()
            meta = data.get("meta", dict(self._DEFAULT_META))
            meta.update(kwargs)
            data["meta"] = meta
            self._write(data)

    def get_meta(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            return deepcopy(data.get("meta", dict(self._DEFAULT_META)))


# Singleton-style store used by monitor and UI
live_state = LiveStateStore()