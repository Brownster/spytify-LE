"""Structured per-track recording history (capped JSONL).

The recorder appends one :class:`TrackResult` per finished track (saved, skipped,
or failed) so the web UI can show a Recorded Tracks table with the year/genre that
were actually tagged — making LastFM mistakes easy to spot. See
``docs/track-history-design.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1

# Outcome values
SAVED = "saved"
SKIPPED_INCOMPLETE = "skipped_incomplete"
SKIPPED_EXISTS = "skipped_exists"
FAILED = "failed"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a stable Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class TrackResult:
    """One recording outcome for the history log."""

    outcome: str
    artist: str = ""
    title: str = ""
    album: str = ""
    track_number: Optional[int] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    path: Optional[str] = None
    duration_ms: int = 0
    reason: Optional[str] = None
    ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if not data.get("ts"):
            data["ts"] = utc_now_iso()
        data["schema_version"] = SCHEMA_VERSION
        return data


class TrackHistoryWriter:
    """Append track results to a capped JSONL file (newest entries retained).

    Thread-safe: outcomes arrive from both the segment thread (incomplete skips)
    and the export worker (saved / already-exists / failed).
    """

    def __init__(self, path: Path | str, cap: int = 500) -> None:
        self.path = Path(path)
        self.cap = cap
        self._lock = threading.Lock()

    def append(self, result: TrackResult) -> None:
        line = json.dumps(result.to_dict(), sort_keys=True)
        with self._lock:
            try:
                lines = self._read_lines()
                lines.append(line)
                if len(lines) > self.cap:
                    lines = lines[-self.cap:]
                self._atomic_write(lines)
            except Exception:
                # History is best-effort; never let it disrupt recording.
                pass

    def read(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return parsed records, newest first (up to ``limit``)."""
        with self._lock:
            lines = self._read_lines()
        records: List[Dict[str, Any]] = []
        for raw in reversed(lines):
            try:
                records.append(json.loads(raw))
            except Exception:
                continue
            if limit is not None and len(records) >= limit:
                break
        return records

    # Internal helpers -------------------------------------------------------
    def _read_lines(self) -> List[str]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            return []

    def _atomic_write(self, lines: List[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write("\n".join(lines) + ("\n" if lines else ""))
                tmp.flush()
            os.replace(tmp_path, self.path)
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
