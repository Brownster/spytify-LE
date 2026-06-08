"""Tests for recorder engine seam types."""

import io
from pathlib import Path
import threading

import pytest

from spotify_splitter.engine import (
    RecorderConfigError,
    RecorderEngine,
    RecorderEngineConfig,
    RecorderError,
)
from spotify_splitter.util import StreamInfo


def make_config(**overrides):
    values = {
        "stream_info": StreamInfo("monitor", 44100, 2),
        "output_dir": Path("/tmp/out"),
        "fmt": "mp3",
        "player": "spotify",
        "dump_metadata": False,
        "queue_size": 200,
        "blocksize": 2048,
        "latency": 0.1,
        "enable_adaptive": True,
        "enable_monitoring": True,
        "enable_metrics": False,
        "debug_mode": False,
        "min_buffer_size": 50,
        "max_buffer_size": 1000,
    }
    values.update(overrides)
    return RecorderEngineConfig(**values)


def test_engine_config_accepts_resolved_values():
    config = make_config(
        playlist_path=Path("/tmp/session.m3u"),
        bundle_playlist=True,
        status_file=Path("/tmp/status.json"),
        control_stdin=True,
    )

    assert config.stream_info.monitor_name == "monitor"
    assert config.output_dir == Path("/tmp/out")
    assert config.playlist_path == Path("/tmp/session.m3u")
    assert config.bundle_playlist is True
    assert config.control_stdin is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("queue_size", 0),
        ("blocksize", 0),
        ("latency", 0),
        ("min_buffer_size", 0),
    ],
)
def test_engine_config_rejects_non_positive_values(field, value):
    with pytest.raises(RecorderConfigError):
        make_config(**{field: value})


def test_engine_config_rejects_invalid_buffer_bounds():
    with pytest.raises(RecorderConfigError, match="max_buffer_size"):
        make_config(min_buffer_size=100, max_buffer_size=50)


def test_engine_config_requires_playlist_for_bundle():
    with pytest.raises(RecorderConfigError, match="bundle_playlist"):
        make_config(bundle_playlist=True)


class FakeManager:
    def __init__(self):
        self.shutdown_calls = 0
        self.flush_calls = 0

    def shutdown_cleanup(self):
        self.shutdown_calls += 1

    def flush_cache(self):
        self.flush_calls += 1


class FakeThread:
    def __init__(self):
        self.join_calls = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls += 1


class BlockingManager:
    def __init__(self):
        self.entered_shutdown = threading.Event()
        self.release_shutdown = threading.Event()
        self.shutdown_calls = 0
        self.flush_calls = 0

    def shutdown_cleanup(self):
        self.shutdown_calls += 1
        self.entered_shutdown.set()
        self.release_shutdown.wait(timeout=1.0)

    def flush_cache(self):
        self.flush_calls += 1


class FakeStream:
    def __init__(self):
        self.enter_calls = 0
        self.exit_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls += 1


def test_engine_creates_runtime_queues_from_config():
    engine = RecorderEngine(make_config(queue_size=123))

    assert engine.audio_queue.maxsize == 123
    assert engine.event_queue.empty()


def test_engine_creates_and_starts_processing_thread():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    ran = threading.Event()

    thread = engine.create_processing_thread(manager, ran.set)

    assert thread.daemon is True
    assert not engine.processing_is_alive()

    engine.start_processing()
    engine.wait_processing(timeout=1.0)

    assert ran.is_set()
    assert not engine.processing_is_alive()


def test_engine_start_enters_stream_and_stop_exits_it():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    stream = FakeStream()
    release = threading.Event()
    started = threading.Event()

    def target():
        started.set()
        release.wait(timeout=1.0)

    engine.start(
        manager=manager,
        processing_target=target,
        stream_factory=lambda: stream,
    )
    assert started.wait(timeout=1.0)

    release.set()
    engine.stop()

    assert stream.enter_calls == 1
    assert stream.exit_calls == 1
    assert manager.shutdown_calls == 1
    assert manager.flush_calls == 1
    assert engine.is_stopped() is True


def test_engine_start_unwinds_stream_on_partial_startup_failure(monkeypatch):
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    stream = FakeStream()

    def fail_start_processing():
        raise RecorderError("processing start failed")

    monkeypatch.setattr(engine, "start_processing", fail_start_processing)

    with pytest.raises(RecorderError, match="processing start failed"):
        engine.start(
            manager=manager,
            processing_target=lambda: None,
            stream_factory=lambda: stream,
        )

    assert stream.enter_calls == 1
    assert stream.exit_calls == 1
    assert manager.shutdown_calls == 1
    assert manager.flush_calls == 1
    assert engine.is_stopped() is True


def test_engine_running_predicates_track_processing_and_cleanup():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    release = threading.Event()
    started = threading.Event()

    def target():
        started.set()
        release.wait(timeout=1.0)

    engine.create_processing_thread(manager, target)

    assert engine.is_running() is False
    assert engine.is_stopped() is False

    engine.start_processing()
    assert started.wait(timeout=1.0)

    assert engine.is_running() is True
    assert engine.is_stopped() is False

    release.set()
    engine.wait_processing(timeout=1.0)

    assert engine.processing_is_alive() is False
    assert engine.is_running() is False
    assert engine.is_stopped() is False


def test_engine_stopped_predicate_tracks_cleanup():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    engine.attach_segment_manager(manager, FakeThread())

    assert engine.is_stopped() is False

    engine.stop()

    assert engine.is_stopped() is True
    assert engine.is_running() is False


def test_engine_stopped_predicate_does_not_block_during_cleanup():
    engine = RecorderEngine(make_config())
    manager = BlockingManager()
    thread = FakeThread()
    engine.attach_segment_manager(manager, thread)

    stopper = threading.Thread(target=engine.stop)
    stopper.start()
    assert manager.entered_shutdown.wait(timeout=1.0)

    assert engine.is_stopped() is False

    manager.release_shutdown.set()
    stopper.join(timeout=1.0)

    assert not stopper.is_alive()
    assert engine.is_stopped() is True
    assert manager.shutdown_calls == 1
    assert manager.flush_calls == 1


def test_engine_timer_is_disabled_without_duration():
    engine = RecorderEngine(make_config(timer_duration_seconds=None))

    assert engine.is_timer_enabled() is False
    snapshot = engine.start_timer(now=10.0)
    tick = engine.tick_timer(now=20.0)

    assert snapshot.enabled is False
    assert tick.snapshot.enabled is False
    assert tick.elapsed_changed is False
    assert tick.expired is False


def test_engine_timer_tracks_elapsed_remaining_and_expiry():
    engine = RecorderEngine(make_config(timer_duration_seconds=5))

    snapshot = engine.start_timer(now=100.0)
    assert snapshot.enabled is True
    assert snapshot.duration_seconds == 5
    assert snapshot.elapsed_seconds == 0
    assert snapshot.remaining_seconds == 5
    assert snapshot.expired is False

    tick = engine.tick_timer(now=102.2)
    assert tick.elapsed_changed is True
    assert tick.expired is False
    assert tick.snapshot.elapsed_seconds == 2
    assert tick.snapshot.remaining_seconds == 3

    same_second_tick = engine.tick_timer(now=102.8)
    assert same_second_tick.elapsed_changed is False
    assert same_second_tick.expired is False
    assert same_second_tick.snapshot.elapsed_seconds == 2
    assert same_second_tick.snapshot.remaining_seconds == 3

    expired_tick = engine.tick_timer(now=105.0)
    assert expired_tick.elapsed_changed is True
    assert expired_tick.expired is True
    assert expired_tick.snapshot.elapsed_seconds == 5
    assert expired_tick.snapshot.remaining_seconds == 0
    assert expired_tick.snapshot.expired is True


def test_engine_lifecycle_stops_when_processing_exits():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    engine.create_processing_thread(manager, lambda: None)
    engine.start_processing()
    engine.wait_processing(timeout=1.0)

    keep_running = engine.run_lifecycle_once()

    assert keep_running is False
    assert engine.is_stopped() is True
    assert manager.shutdown_calls == 1
    assert manager.flush_calls == 1


def test_engine_lifecycle_stops_on_timer_expiry():
    statuses = []
    timer_ticks = []
    timer_expiries = []
    engine = RecorderEngine(
        make_config(timer_duration_seconds=1),
        status_publisher=statuses.append,
    )
    manager = FakeManager()
    engine.attach_segment_manager(manager, FakeThread())
    engine.start_timer(now=-1000.0)

    keep_running = engine.run_lifecycle_once(
        on_timer_tick=timer_ticks.append,
        on_timer_expired=timer_expiries.append,
    )

    assert keep_running is False
    assert engine.is_stopped() is True
    assert manager.shutdown_calls == 1
    assert manager.flush_calls == 1
    assert timer_expiries
    assert timer_expiries[0].expired is True
    assert statuses == ["stopping", "stopped"]


def test_engine_start_processing_requires_configured_thread():
    engine = RecorderEngine(make_config())

    with pytest.raises(RecorderError, match="processing thread"):
        engine.start_processing()


def test_engine_stop_flushes_once_and_publishes_status():
    statuses = []
    engine = RecorderEngine(make_config(), status_publisher=statuses.append)
    manager = FakeManager()
    thread = FakeThread()
    engine.attach_segment_manager(manager, thread)

    engine.stop()
    engine.stop()

    assert manager.shutdown_calls == 1
    assert manager.flush_calls == 1
    assert thread.join_calls == 1
    assert engine.event_queue.get_nowait() == ("shutdown", None)
    assert statuses == ["stopping", "stopped"]


def test_engine_stop_can_skip_flush():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    engine.attach_segment_manager(manager, FakeThread())

    engine.stop(flush=False)

    assert manager.shutdown_calls == 0
    assert manager.flush_calls == 1


def test_engine_handle_stop_command_requests_reader_exit():
    statuses = []
    engine = RecorderEngine(make_config(), status_publisher=statuses.append)
    manager = FakeManager()
    engine.attach_segment_manager(manager, FakeThread())

    should_exit = engine.handle_command({"cmd": "stop", "flush": True})

    assert should_exit is True
    assert engine.control_stop_requested.is_set()
    assert manager.shutdown_calls == 1
    assert statuses == ["stopping", "stopped"]


def test_engine_handle_unknown_command_keeps_reader_running():
    engine = RecorderEngine(make_config())

    should_exit = engine.handle_command({"cmd": "pause"})

    assert should_exit is False
    assert not engine.control_stop_requested.is_set()


def test_engine_control_reader_ignores_invalid_input_before_stop():
    statuses = []
    stop_callbacks = []
    engine = RecorderEngine(make_config(), status_publisher=statuses.append)
    manager = FakeManager()
    engine.attach_segment_manager(manager, FakeThread())
    stream = io.StringIO(
        "\n"
        "not json\n"
        "[]\n"
        '{"cmd":"pause"}\n'
        '{"cmd":"stop","flush":true}\n'
        '{"cmd":"stop","flush":true}\n'
    )

    engine.read_control_stream(stream, on_stop_requested=lambda: stop_callbacks.append("stop"))

    assert stop_callbacks == ["stop"]
    assert engine.control_stop_requested.is_set()
    assert manager.shutdown_calls == 1
    assert statuses == ["stopping", "stopped"]


def test_engine_start_control_reader_runs_on_daemon_thread():
    engine = RecorderEngine(make_config())
    manager = FakeManager()
    engine.attach_segment_manager(manager, FakeThread())

    thread = engine.start_control_reader(io.StringIO('{"cmd":"stop","flush":true}\n'))
    engine.wait_control_reader(timeout=1.0)

    assert thread.daemon is True
    assert not thread.is_alive()
    assert engine.control_stop_requested.is_set()
