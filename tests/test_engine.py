"""Tests for recorder engine seam types."""

from pathlib import Path

import pytest

from spotify_splitter.engine import RecorderConfigError, RecorderEngineConfig
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
