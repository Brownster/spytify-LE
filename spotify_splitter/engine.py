"""Recorder engine seam definitions.

This module starts with the stable types needed to extract orchestration from
``main.py``. Pipeline ownership moves here in follow-up Pass 1 slices.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

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
class TimerSnapshot:
    """Current engine timer facts."""

    enabled: bool
    duration_seconds: int = 0
    elapsed_seconds: int = 0
    remaining_seconds: int = 0
    expired: bool = False


@dataclass(frozen=True)
class TimerTick:
    """Timer tick result for frontend/status updates."""

    snapshot: TimerSnapshot
    elapsed_changed: bool = False
    expired: bool = False


class SegmentManagerLike(Protocol):
    """Subset of SegmentManager used by the engine shutdown path."""

    def shutdown_cleanup(self) -> None: ...

    def flush_cache(self) -> None: ...


StatusPublisher = Callable[[Optional[str]], None]
ThreadTarget = Callable[[], None]
ControlStopCallback = Callable[[], None]
HeartbeatCallback = Callable[[], None]
TimerCallback = Callable[[TimerTick], None]


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


class RecorderEngine:
    """Recorder orchestration shell.

    This class intentionally starts as a narrow owner for runtime queues and the
    shared cleanup/control path. Stream and MPRIS startup move here in follow-up
    slices.
    """

    def __init__(
        self,
        config: RecorderEngineConfig,
        status_publisher: Optional[StatusPublisher] = None,
    ) -> None:
        self.config = config
        self.audio_queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.event_queue: queue.Queue = queue.Queue()
        self.control_stop_requested = threading.Event()

        self._status_publisher = status_publisher
        self._segment_manager: Optional[SegmentManagerLike] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None
        self._lifecycle_thread: Optional[threading.Thread] = None
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()
        self._stopped = threading.Event()
        self._timer_duration_seconds = config.timer_duration_seconds or 0
        self._timer_start: Optional[float] = None
        self._timer_elapsed_seconds = 0
        self._timer_remaining_seconds = self._timer_duration_seconds

    def set_status_publisher(self, status_publisher: StatusPublisher) -> None:
        self._status_publisher = status_publisher

    def attach_segment_manager(
        self,
        manager: SegmentManagerLike,
        processing_thread: threading.Thread,
    ) -> None:
        self._segment_manager = manager
        self._processing_thread = processing_thread

    def create_processing_thread(
        self,
        manager: SegmentManagerLike,
        target: ThreadTarget,
    ) -> threading.Thread:
        self._segment_manager = manager
        self._processing_thread = threading.Thread(target=target, daemon=True)
        return self._processing_thread

    def start_processing(self) -> None:
        if not self._processing_thread:
            raise RecorderError("processing thread has not been configured")
        self._processing_thread.start()

    def processing_is_alive(self) -> bool:
        return bool(self._processing_thread and self._processing_thread.is_alive())

    def is_running(self) -> bool:
        """Return true only while recorder work is actively running."""
        return self.processing_is_alive() and not self.is_stopped()

    def is_stopped(self) -> bool:
        """Return true after guarded cleanup completes."""
        return self._stopped.is_set()

    def is_timer_enabled(self) -> bool:
        return self._timer_duration_seconds > 0

    def start_timer(self, now: Optional[float] = None) -> TimerSnapshot:
        if not self.is_timer_enabled():
            return self.timer_snapshot()
        self._timer_start = time.monotonic() if now is None else now
        self._timer_elapsed_seconds = 0
        self._timer_remaining_seconds = self._timer_duration_seconds
        return self.timer_snapshot()

    def tick_timer(self, now: Optional[float] = None) -> TimerTick:
        if not self.is_timer_enabled() or self._timer_start is None:
            return TimerTick(self.timer_snapshot())

        current_time = time.monotonic() if now is None else now
        elapsed = max(0.0, current_time - self._timer_start)
        elapsed_seconds = int(elapsed)
        remaining_seconds = max(0, self._timer_duration_seconds - elapsed_seconds)
        elapsed_changed = elapsed_seconds != self._timer_elapsed_seconds

        self._timer_elapsed_seconds = elapsed_seconds
        self._timer_remaining_seconds = remaining_seconds
        expired = elapsed >= self._timer_duration_seconds
        return TimerTick(self.timer_snapshot(expired=expired), elapsed_changed, expired)

    def timer_snapshot(self, expired: bool = False) -> TimerSnapshot:
        return TimerSnapshot(
            enabled=self.is_timer_enabled(),
            duration_seconds=self._timer_duration_seconds,
            elapsed_seconds=self._timer_elapsed_seconds,
            remaining_seconds=self._timer_remaining_seconds,
            expired=expired,
        )

    def wait_processing(self, timeout: Optional[float] = None) -> None:
        if self._processing_thread:
            self._processing_thread.join(timeout=timeout)

    def start_lifecycle_loop(
        self,
        heartbeat_interval: float,
        on_heartbeat: Optional[HeartbeatCallback] = None,
        on_timer_tick: Optional[TimerCallback] = None,
        on_timer_expired: Optional[TimerCallback] = None,
        poll_interval: float = 0.1,
    ) -> threading.Thread:
        self._lifecycle_thread = threading.Thread(
            target=self._lifecycle_loop,
            args=(
                heartbeat_interval,
                on_heartbeat,
                on_timer_tick,
                on_timer_expired,
                poll_interval,
            ),
            daemon=True,
        )
        self._lifecycle_thread.start()
        return self._lifecycle_thread

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._stopped.wait(timeout=timeout)

    def run_lifecycle_once(
        self,
        on_timer_tick: Optional[TimerCallback] = None,
        on_timer_expired: Optional[TimerCallback] = None,
    ) -> bool:
        """Run one lifecycle iteration. Return false when the loop should exit."""
        if self.is_stopped():
            return False

        if not self.processing_is_alive():
            self.stop(flush=True)
            return False

        if self.is_timer_enabled():
            timer_tick = self.tick_timer()
            if timer_tick.elapsed_changed and on_timer_tick:
                on_timer_tick(timer_tick)
            if timer_tick.expired:
                if on_timer_expired:
                    on_timer_expired(timer_tick)
                self.stop(flush=True)
                return False

        return True

    def _lifecycle_loop(
        self,
        heartbeat_interval: float,
        on_heartbeat: Optional[HeartbeatCallback],
        on_timer_tick: Optional[TimerCallback],
        on_timer_expired: Optional[TimerCallback],
        poll_interval: float,
    ) -> None:
        last_heartbeat = time.monotonic()
        while self.run_lifecycle_once(on_timer_tick, on_timer_expired):
            now = time.monotonic()
            if on_heartbeat and heartbeat_interval > 0 and now - last_heartbeat >= heartbeat_interval:
                on_heartbeat()
                last_heartbeat = now
            time.sleep(poll_interval)

    def start_control_reader(
        self,
        input_stream: Iterable[str],
        on_stop_requested: Optional[ControlStopCallback] = None,
    ) -> threading.Thread:
        self._control_thread = threading.Thread(
            target=self.read_control_stream,
            args=(input_stream, on_stop_requested),
            daemon=True,
        )
        self._control_thread.start()
        return self._control_thread

    def wait_control_reader(self, timeout: Optional[float] = None) -> None:
        if self._control_thread:
            self._control_thread.join(timeout=timeout)

    def read_control_stream(
        self,
        input_stream: Iterable[str],
        on_stop_requested: Optional[ControlStopCallback] = None,
    ) -> None:
        for line in input_stream:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning("Invalid stdin control command: %s", e)
                continue
            if not isinstance(command, dict):
                logging.warning("Invalid stdin control command type: %s", type(command).__name__)
                continue
            if command.get("cmd") == "stop" and on_stop_requested:
                on_stop_requested()
            if self.handle_command(command):
                return

    def stop(self, flush: bool = True) -> None:
        with self._cleanup_lock:
            if self._cleanup_done:
                return

            logging.info("Processing remaining tracks before shutdown...")
            if flush and self._segment_manager:
                self._segment_manager.shutdown_cleanup()
            self.event_queue.put(("shutdown", None))
            if self.processing_is_alive():
                self.wait_processing()
            if self._segment_manager:
                self._segment_manager.flush_cache()
            self._cleanup_done = True
            self._stopped.set()
            self._publish_status("stopped")

    def handle_command(self, command: dict) -> bool:
        cmd = command.get("cmd")
        if cmd == "stop":
            logging.info("Received stdin control command: stop")
            self._publish_status("stopping")
            self.stop(flush=bool(command.get("flush", True)))
            self.control_stop_requested.set()
            return True

        logging.warning("Ignoring unsupported stdin control command: %s", cmd)
        return False

    def _publish_status(self, state: str) -> None:
        if self._status_publisher:
            self._status_publisher(state)
