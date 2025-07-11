import json
import subprocess
import pytest
import sys
import types

try:
    import sounddevice
except Exception:  # pragma: no cover - sounddevice may not be installed
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.check_input_settings = lambda device: None
    sys.modules["sounddevice"] = sounddevice
from spotify_splitter.util import get_spotify_stream_info, _is_spotify, StreamInfo


def test_get_spotify_stream_info(monkeypatch):
    inputs = json.dumps(
        [
            {
                "properties": {"application.name": "Spotify"},
                "sink": 5,
                "sample_spec": {"rate": 48000, "channels": 2},
            }
        ]
    ).encode()
    sinks = json.dumps(
        [{"index": 5, "monitor_source_name": "alsa_output.monitor"}]
    ).encode()

    def fake_cmd(cmd):
        if "sink-inputs" in cmd:
            return inputs
        return sinks

    monkeypatch.setattr(subprocess, "check_output", lambda cmd: fake_cmd(cmd))
    monkeypatch.setattr(sounddevice, "check_input_settings", lambda device: None)
    info = get_spotify_stream_info()
    assert info == StreamInfo("alsa_output.monitor", 48000, 2)


def test_get_spotify_stream_info_new_format(monkeypatch):
    """Handle newer pactl JSON without monitor_source_name and string spec."""
    inputs = json.dumps(
        [
            {
                "properties": {"application.name": "Spotify"},
                "sink": 7,
                "sample_specification": "s16le 2ch 44100Hz",
            }
        ]
    ).encode()
    sinks = json.dumps(
        [{"index": 7, "name": "alsa_output.pci-0000_00_1f.3.analog-stereo"}]
    ).encode()

    def fake_cmd(cmd):
        if "sink-inputs" in cmd:
            return inputs
        return sinks

    monkeypatch.setattr(subprocess, "check_output", lambda cmd: fake_cmd(cmd))
    monkeypatch.setattr(sounddevice, "check_input_settings", lambda device: None)
    info = get_spotify_stream_info()
    assert info == StreamInfo(
        "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor", 44100, 2
    )


def test_get_spotify_stream_info_node_name(monkeypatch):
    """Use node.name when no monitor_source_name is available."""
    inputs = json.dumps(
        [
            {
                "properties": {"application.name": "Spotify", "node.name": "spotify"},
                "sink": 8,
                "sample_spec": {"rate": 44100, "channels": 2},
            }
        ]
    ).encode()
    sinks = json.dumps([{"index": 8}]).encode()

    def fake_cmd(cmd):
        if "sink-inputs" in cmd:
            return inputs
        return sinks

    monkeypatch.setattr(subprocess, "check_output", lambda cmd: fake_cmd(cmd))
    monkeypatch.setattr(sounddevice, "check_input_settings", lambda device: None)
    info = get_spotify_stream_info()
    assert info == StreamInfo("spotify", 44100, 2)


def test_get_spotify_stream_info_sink_rate(monkeypatch):
    """Sink sample rate overrides input spec."""
    inputs = json.dumps(
        [
            {
                "properties": {"application.name": "Spotify"},
                "sink": 9,
                "sample_spec": {"rate": 44100, "channels": 2},
            }
        ]
    ).encode()
    sinks = json.dumps(
        [
            {
                "index": 9,
                "monitor_source_name": "alsa_output.monitor",
                "sample_spec": {"rate": 24000, "channels": 2},
            }
        ]
    ).encode()

    def fake_cmd(cmd):
        if "sink-inputs" in cmd:
            return inputs
        return sinks

    monkeypatch.setattr(subprocess, "check_output", lambda cmd: fake_cmd(cmd))
    monkeypatch.setattr(sounddevice, "check_input_settings", lambda device: None)
    info = get_spotify_stream_info()
    assert info == StreamInfo("alsa_output.monitor", 24000, 2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("application.name", "spotify"),
        ("application.icon_name", "com.spotify.Client"),
        ("application.process.binary", "spotify"),
        ("pipewire.access.portal.app_id", "com.spotify.Client"),
        ("media.name", "Spotify"),
    ],
)
def test_is_spotify(monkeypatch, key, value):
    props = {key: value}
    assert _is_spotify(props)
