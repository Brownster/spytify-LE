"""Tests for recorder engine seam types."""

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
    assert statuses == ["stopped"]


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
