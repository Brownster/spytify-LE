"""Recorder engine seam definitions.

This module starts with the stable types needed to extract orchestration from
``main.py``. Pipeline ownership moves here in follow-up Pass 1 slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .util import StreamInfo


class RecorderError(Exception):
    """Base exception for recorder engine failures."""


class StreamNotFoundError(RecorderError):
    """Raised when the Spotify monitor source cannot be found."""


class RecorderConfigError(RecorderError):
    """Raised when engine configuration is invalid."""


class RecorderDbusError(RecorderError):
    """Raised when MPRIS/D-Bus setup fails."""


@dataclass(frozen=True)
class RecorderEngineConfig:
    """Resolved configuration for a recorder engine run."""

    stream_info: StreamInfo
    output_dir: Path
    fmt: str
    player: str
    dump_metadata: bool
    queue_size: int
    blocksize: int
    latency: float
    enable_adaptive: bool
    enable_monitoring: bool
    enable_metrics: bool
    debug_mode: bool
    min_buffer_size: int
    max_buffer_size: int
    playlist_path: Optional[Path] = None
    bundle_playlist: bool = False
    bundle_album_art_uri: Optional[str] = None
    playlist_base_path: Optional[str] = None
    max_duration: Optional[str] = None
    timer_duration_seconds: Optional[int] = None
    allow_overwrite: bool = False
    lastfm_api_key: Optional[str] = None
    status_file: Optional[Path] = None
    control_stdin: bool = False

    def __post_init__(self) -> None:
        if self.queue_size <= 0:
            raise RecorderConfigError("queue_size must be positive")
        if self.blocksize <= 0:
            raise RecorderConfigError("blocksize must be positive")
        if self.latency <= 0:
            raise RecorderConfigError("latency must be positive")
        if self.min_buffer_size <= 0:
            raise RecorderConfigError("min_buffer_size must be positive")
        if self.max_buffer_size < self.min_buffer_size:
            raise RecorderConfigError("max_buffer_size must be greater than or equal to min_buffer_size")
        if self.bundle_playlist and not self.playlist_path:
            raise RecorderConfigError("bundle_playlist requires playlist_path")
