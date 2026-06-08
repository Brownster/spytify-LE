"""Recorder engine seam definitions.

This module contains the frontend-neutral recorder orchestration being
extracted from ``main.py`` during Pass 1.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Callable, Iterable, Optional, Protocol, Type

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


class AudioStreamLike(Protocol):
    """Context-manager surface used by the engine-owned stream lifecycle."""

    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> object: ...


StatusPublisher = Callable[[Optional[str]], None]
ThreadTarget = Callable[[], None]
ControlStopCallback = Callable[[], None]
HeartbeatCallback = Callable[[], None]
TimerCallback = Callable[[TimerTick], None]
AudioStreamFactory = Callable[[], AudioStreamLike]
TrackEventRunner = Callable[[Callable[[object], None], Callable[[str], None]], None]
TagOutputCallback = Callable[[Path, Optional[Path]], None]


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

    The engine owns runtime queues, stream lifetime, worker thread startup, and
    the shared cleanup/control path. Pipeline internals remain in audio.py and
    segmenter.py until the Pass 2 hot-path work.
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
        self._mpris_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None
        self._lifecycle_thread: Optional[threading.Thread] = None
        self._audio_stream: Optional[AudioStreamLike] = None
        self._audio_stream_entered = False
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()
        self._finalize_done = False
        self._finalize_lock = threading.Lock()
        self._stopped = threading.Event()
        self._timer_duration_seconds = config.timer_duration_seconds or 0
        self._timer_start: Optional[float] = None
        self._timer_elapsed_seconds = 0
        self._timer_remaining_seconds = self._timer_duration_seconds
        self._metrics_collector: Optional[object] = None
        self._performance_dashboard: Optional[object] = None
        self._performance_optimizer: Optional[object] = None
        self._tag_output: Optional[TagOutputCallback] = None
        self._final_diagnostics_enabled = False

    def set_status_publisher(self, status_publisher: StatusPublisher) -> None:
        self._status_publisher = status_publisher

    def configure_post_run_cleanup(
        self,
        metrics_collector: Optional[object] = None,
        performance_dashboard: Optional[object] = None,
        performance_optimizer: Optional[object] = None,
        tag_output: Optional[TagOutputCallback] = None,
        final_diagnostics: bool = False,
    ) -> None:
        self._metrics_collector = metrics_collector
        self._performance_dashboard = performance_dashboard
        self._performance_optimizer = performance_optimizer
        self._tag_output = tag_output
        self._final_diagnostics_enabled = final_diagnostics

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

    def start(
        self,
        manager: SegmentManagerLike,
        processing_target: ThreadTarget,
        stream_factory: AudioStreamFactory,
        track_event_runner: Optional[TrackEventRunner] = None,
        on_track_change: Optional[Callable[[object], None]] = None,
        on_playback_status: Optional[Callable[[str], None]] = None,
        health_monitor_target: Optional[ThreadTarget] = None,
        control_input_stream: Optional[Iterable[str]] = None,
        on_control_stop_requested: Optional[ControlStopCallback] = None,
        heartbeat_interval: float = 0.0,
        on_heartbeat: Optional[HeartbeatCallback] = None,
        on_timer_tick: Optional[TimerCallback] = None,
        on_timer_expired: Optional[TimerCallback] = None,
    ) -> None:
        """Start recorder-owned runtime resources and return immediately."""
        if self._processing_thread and self._processing_thread.is_alive():
            raise RecorderError("recorder engine is already running")

        try:
            self._segment_manager = manager
            self._audio_stream = stream_factory()
            self._audio_stream.__enter__()
            self._audio_stream_entered = True

            self.create_processing_thread(manager, processing_target)
            self.start_processing()

            if health_monitor_target:
                self._health_thread = threading.Thread(target=health_monitor_target, daemon=True)
                self._health_thread.start()

            if track_event_runner:
                self._mpris_thread = threading.Thread(
                    target=self._run_track_events,
                    args=(track_event_runner, on_track_change, on_playback_status),
                    daemon=True,
                )
                self._mpris_thread.start()
                logging.info("MPRIS thread started")

            if self.config.control_stdin and control_input_stream is not None:
                self.start_control_reader(control_input_stream, on_control_stop_requested)

            if self.is_timer_enabled():
                self.start_timer()

            self.start_lifecycle_loop(
                heartbeat_interval=heartbeat_interval,
                on_heartbeat=on_heartbeat,
                on_timer_tick=on_timer_tick,
                on_timer_expired=on_timer_expired,
            )
        except Exception as e:
            logging.error("Recorder startup failed: %s", e)
            try:
                self.stop(flush=True)
            except Exception as cleanup_error:
                logging.debug("Error while unwinding failed recorder startup: %s", cleanup_error)
            if isinstance(e, RecorderError):
                raise
            raise RecorderError("Recorder startup failed") from e

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

    def run(self, *args, **kwargs) -> None:
        """Run a headless recorder session through finalization."""
        try:
            self.start(*args, **kwargs)
            self.wait()
        finally:
            self.finalize_post_run()

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

            self._publish_status("stopping")
            logging.info("Processing remaining tracks before shutdown...")
            if flush and self._segment_manager:
                self._segment_manager.shutdown_cleanup()
            self.event_queue.put(("shutdown", None))
            if self.processing_is_alive():
                self.wait_processing()
            if self._segment_manager:
                self._segment_manager.flush_cache()
            self._exit_audio_stream()
            self._cleanup_done = True
            self._stopped.set()
            self._publish_status("stopped")

    def finalize_post_run(self) -> None:
        """Run non-realtime cleanup after recording has stopped."""
        with self._finalize_lock:
            if self._finalize_done:
                return
            self._stop_performance_optimizer()
            self._stop_performance_dashboard()
            self._stop_metrics_collection()
            self._close_playlist()
            self._run_tagger()
            self._finalize_done = True

    def handle_command(self, command: dict) -> bool:
        cmd = command.get("cmd")
        if cmd == "stop":
            logging.info("Received stdin control command: stop")
            self.stop(flush=bool(command.get("flush", True)))
            self.control_stop_requested.set()
            return True

        logging.warning("Ignoring unsupported stdin control command: %s", cmd)
        return False

    def _publish_status(self, state: str) -> None:
        if self._status_publisher:
            self._status_publisher(state)

    def _run_track_events(
        self,
        track_event_runner: TrackEventRunner,
        on_track_change: Optional[Callable[[object], None]],
        on_playback_status: Optional[Callable[[str], None]],
    ) -> None:
        try:
            logging.info("Starting MPRIS tracking for player: %s", self.config.player)
            track_event_runner(
                on_track_change or (lambda _track: None),
                on_playback_status or (lambda _status: None),
            )
        except KeyboardInterrupt:
            logging.debug("MPRIS tracking interrupted")
        except Exception as e:
            logging.error("MPRIS tracking failed: %s", e)

    def _exit_audio_stream(self) -> None:
        if not self._audio_stream_entered or not self._audio_stream:
            return
        try:
            self._audio_stream.__exit__(None, None, None)
        except Exception as e:
            logging.debug("Error stopping audio stream: %s", e)
        finally:
            self._audio_stream_entered = False

    def _stop_performance_optimizer(self) -> None:
        if not self._performance_optimizer:
            return
        try:
            self._performance_optimizer.stop_optimization()
            logging.info("Performance optimizer stopped")
        except Exception as e:
            logging.error("Error stopping performance optimizer: %s", e)

    def _stop_performance_dashboard(self) -> None:
        if not self._performance_dashboard:
            return
        try:
            self._performance_dashboard.stop_monitoring()
            logging.info("Performance dashboard stopped")
        except Exception as e:
            logging.error("Error stopping performance dashboard: %s", e)

    def _stop_metrics_collection(self) -> None:
        if not self._metrics_collector:
            return
        try:
            self._metrics_collector.stop_collection()
            logging.info("Metrics collection stopped")
            if self._final_diagnostics_enabled:
                self._log_final_diagnostics()
        except Exception as e:
            logging.error("Error stopping metrics collection: %s", e)

    def _log_final_diagnostics(self) -> None:
        if not self._metrics_collector:
            return
        try:
            report = self._metrics_collector.generate_diagnostic_report()
            logging.info("Session performance summary:")
            logging.info("  - Total metrics collected: %s", report.summary.get("total_metrics", 0))
            logging.info(
                "  - Collection uptime: %.1fs",
                report.summary.get("collection_uptime_seconds", 0),
            )
            if report.recommendations:
                logging.info("  - Recommendations:")
                for recommendation in report.recommendations[:3]:
                    logging.info("    * %s", recommendation)

            if self._performance_optimizer:
                suggestions = self._performance_optimizer.get_optimization_suggestions(limit=3)
                if suggestions:
                    logging.info("  - Performance optimization suggestions:")
                    for suggestion in suggestions:
                        logging.info("    * %s: %s", suggestion.title, suggestion.description)
        except Exception as e:
            logging.debug("Error generating final report: %s", e)

    def _close_playlist(self) -> None:
        if not self._segment_manager or not hasattr(self._segment_manager, "close_playlist"):
            return
        try:
            self._segment_manager.close_playlist()
        except Exception as e:
            logging.debug("Error closing playlist: %s", e)

    def _run_tagger(self) -> None:
        if not self._tag_output:
            return
        try:
            playlist_path = self.config.playlist_path.resolve() if self.config.playlist_path else None
            self._tag_output(self.config.output_dir, playlist_path)
        except Exception as e:
            logging.debug("Error calling tagger API: %s", e)
