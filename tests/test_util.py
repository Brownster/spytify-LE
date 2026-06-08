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
from spotify_splitter.util import (
    get_spotify_stream_info,
    stream_info_for_source,
    _is_spotify,
    StreamInfo,
)


def _check_output_for(mapping):
    """Return a fake subprocess.check_output keyed on the pactl/pw-dump command."""

    def fake(cmd):
        if cmd[0] == "pw-dump":
            return mapping.get("pwdump", b"[]")
        if "sink-inputs" in cmd:
            return mapping["inputs"]
        if "sources" in cmd:
            return mapping.get("sources", b"[]")
        return mapping["sinks"]

    return fake


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


def test_pipewire_correlation_falls_back_to_monitor(monkeypatch):
    """Modern flatpak: sink-input props are stripped; correlate via pw-dump node.name."""
    inputs = json.dumps(
        [
            {
                "properties": {"node.name": "audio-src", "media.name": "audio-src"},
                "sink": 61,
                "sample_spec": {"rate": 48000, "channels": 2},
            }
        ]
    ).encode()
    sinks = json.dumps(
        [
            {
                "index": 61,
                "name": "alsa_output.analog-stereo",
                "monitor_source_name": "alsa_output.analog-stereo.monitor",
                "sample_spec": {"rate": 48000, "channels": 2},
            }
        ]
    ).encode()
    pwdump = json.dumps(
        [
            {
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "media.class": "Stream/Output/Audio",
                        "application.name": "spotify",
                        "node.name": "audio-src",
                    }
                },
            }
        ]
    ).encode()

    monkeypatch.setattr(
        subprocess,
        "check_output",
        _check_output_for({"inputs": inputs, "sinks": sinks, "pwdump": pwdump}),
    )

    def fail(device):
        raise ValueError("audio-src is not a PortAudio device here")

    monkeypatch.setattr(sounddevice, "check_input_settings", fail)
    info = get_spotify_stream_info()
    assert info == StreamInfo("alsa_output.analog-stereo.monitor", 48000, 2)


def test_pipewire_correlation_uses_capturable_node(monkeypatch):
    """When the correlated Spotify node is itself a capturable device, use it."""
    inputs = json.dumps(
        [
            {
                "properties": {"node.name": "audio-src", "media.name": "audio-src"},
                "sink": 61,
                "sample_spec": {"rate": 48000, "channels": 2},
            }
        ]
    ).encode()
    sinks = json.dumps([{"index": 61, "sample_spec": {"rate": 48000, "channels": 2}}]).encode()
    pwdump = json.dumps(
        [
            {
                "info": {
                    "props": {
                        "media.class": "Stream/Output/Audio",
                        "application.process.binary": "spotify",
                        "node.name": "audio-src",
                    }
                }
            }
        ]
    ).encode()

    monkeypatch.setattr(
        subprocess,
        "check_output",
        _check_output_for({"inputs": inputs, "sinks": sinks, "pwdump": pwdump}),
    )
    monkeypatch.setattr(sounddevice, "check_input_settings", lambda device: None)
    info = get_spotify_stream_info()
    assert info == StreamInfo("audio-src", 48000, 2)


def test_not_found_without_pipewire_match(monkeypatch):
    """No strict match and no Spotify node in pw-dump -> RuntimeError."""
    inputs = json.dumps([{"properties": {"node.name": "audio-src"}, "sink": 61}]).encode()
    sinks = json.dumps([{"index": 61}]).encode()
    monkeypatch.setattr(
        subprocess,
        "check_output",
        _check_output_for({"inputs": inputs, "sinks": sinks, "pwdump": b"[]"}),
    )
    monkeypatch.setattr(sounddevice, "check_input_settings", lambda device: None)
    with pytest.raises(RuntimeError):
        get_spotify_stream_info()


def test_stream_info_for_source_reads_pactl_spec(monkeypatch):
    """Explicit source override reads the source sample spec from pactl."""
    sources = json.dumps(
        [
            {
                "name": "alsa_output.analog-stereo.monitor",
                "sample_spec": {"rate": 48000, "channels": 2},
            }
        ]
    ).encode()
    monkeypatch.setattr(
        subprocess,
        "check_output",
        _check_output_for({"inputs": b"[]", "sinks": b"[]", "sources": sources}),
    )
    info = stream_info_for_source("alsa_output.analog-stereo.monitor")
    assert info == StreamInfo("alsa_output.analog-stereo.monitor", 48000, 2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("application.name", "spotify"),
        ("application.icon_name", "com.spotify.Client"),
        ("application.process.binary", "spotify"),
        ("pipewire.access.portal.app_id", "com.spotify.Client"),
        ("media.name", "Spotify"),
        # spotifyd/librespot detection
        ("application.name", "librespot"),
        ("application.process.binary", "spotifyd"),
        ("media.name", "Spotify endpoint"),
    ],
)
def test_is_spotify(monkeypatch, key, value):
    props = {key: value}
    assert _is_spotify(props)
