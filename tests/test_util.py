import json
import subprocess
from spotify_splitter.util import find_spotify_monitor


def test_find_spotify_monitor(monkeypatch):
    inputs = json.dumps([
        {"properties": {"application.name": "Spotify"}, "sink": 5}
    ]).encode()
    sinks = json.dumps([
        {"index": 5, "monitor_source_name": "alsa_output.monitor"}
    ]).encode()

    def fake_cmd(cmd):
        if "sink-inputs" in cmd:
            return inputs
        return sinks

    monkeypatch.setattr(subprocess, "check_output", lambda cmd: fake_cmd(cmd))
    assert find_spotify_monitor() == "alsa_output.monitor"
