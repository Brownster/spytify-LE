import json
import subprocess
import pytest
from spotify_splitter.util import get_spotify_stream_info, _is_spotify, StreamInfo


def test_get_spotify_stream_info(monkeypatch):
    inputs = json.dumps([
        {
            "properties": {"application.name": "Spotify"},
            "sink": 5,
            "sample_spec": {"rate": 48000, "channels": 2},
        }
    ]).encode()
    sinks = json.dumps([
        {"index": 5, "monitor_source_name": "alsa_output.monitor"}
    ]).encode()

    def fake_cmd(cmd):
        if "sink-inputs" in cmd:
            return inputs
        return sinks

    monkeypatch.setattr(subprocess, "check_output", lambda cmd: fake_cmd(cmd))
    info = get_spotify_stream_info()
    assert info == StreamInfo("alsa_output.monitor", 48000, 2)


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
