"""Structured recorder status and atomic JSON persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a stable Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class TrackStatus:
    artist: str = ""
    title: str = ""
    album: str = ""
    duration_ms: int = 0
    position: float = 0.0

    @classmethod
    def from_track_info(cls, track: Any) -> "TrackStatus":
        return cls(
            artist=getattr(track, "artist", "") or "",
            title=getattr(track, "title", "") or "",
            album=getattr(track, "album", "") or "",
            duration_ms=int(getattr(track, "duration_ms", 0) or 0),
            position=float(getattr(track, "position", 0.0) or 0.0),
        )


@dataclass
class TimerStatus:
    enabled: bool = False
    elapsed_seconds: int = 0
    remaining_seconds: int = 0


@dataclass
class AudioStatus:
    queue_depth: int = 0
    dropped_frames: int = 0
    buffer_warnings: int = 0


@dataclass
class RecorderStatus:
    state: str = "stopped"
    pid: int = field(default_factory=os.getpid)
    current_track: Optional[TrackStatus] = None
    tracks_recorded: int = 0
    timer: TimerStatus = field(default_factory=TimerStatus)
    audio: AudioStatus = field(default_factory=AudioStatus)
    last_error: Optional[str] = None
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def set_track(self, track: Any) -> None:
        self.current_track = TrackStatus.from_track_info(track)
        self.touch()


class AtomicStatusWriter:
    """Write recorder status as an atomically replaced JSON file."""

    def __init__(self, path: Path | str, fsync: bool = True):
        self.path = Path(path)
        self.fsync = fsync

    def write(self, status: RecorderStatus) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        status.touch()
        payload = status.to_dict()

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
                json.dump(payload, tmp, indent=2, sort_keys=True)
                tmp.write("\n")
                tmp.flush()
                if self.fsync:
                    os.fsync(tmp.fileno())

            os.replace(tmp_path, self.path)
            if self.fsync:
                self._fsync_parent()
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def read(self) -> Dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _fsync_parent(self) -> None:
        try:
            fd = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
