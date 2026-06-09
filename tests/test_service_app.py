"""Tests for the web service recorder supervisor."""

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess

from spoti2_service.service_app import (
    RecorderSupervisor,
    merge_web_config,
    filter_recorder_logs,
)


def test_filter_recorder_logs_shows_saved_track():
    """The save line is 'Saved /path' (no colon) and must appear in the panel."""
    lines = ["INFO: Saved /home/marc/Music/Artist/Album/01 - Track.mp3\n"]
    html = filter_recorder_logs(lines, verbose=False)
    assert "log-success" in html
    assert "01 - Track.mp3" in html
    assert "Waiting for tracks" not in html


def test_filter_recorder_logs_skips_and_errors():
    lines = [
        "INFO: Skipping incomplete track 'X' (captured 5000ms of expected 200000ms)\n",
        "ERROR: Recorder failed\n",
    ]
    html = filter_recorder_logs(lines, verbose=False)
    assert "log-warning" in html  # skip
    assert "log-error" in html


def test_filter_recorder_logs_waiting_when_empty():
    assert "Waiting for tracks" in filter_recorder_logs(["DEBUG: noise\n"], verbose=False)
from spoti2_service.web_ui import render_index
from spotify_splitter.user_config import DEFAULT_CONFIG


def test_merge_web_config_partial_form_preserves_other_fields():
    """A timer-only form must not wipe the LastFM key or other settings."""
    config = {
        "output": "/music",
        "format": "flac",
        "lastfm_api_key": "KEY123",
        "allow_overwrite": True,
        "enable_adaptive": True,
        "playlist": "/p.m3u",
    }
    merged = merge_web_config(config, {"max_duration": ["90m"]})
    assert merged["max_duration"] == "90m"
    assert merged["lastfm_api_key"] == "KEY123"
    assert merged["allow_overwrite"] is True
    assert merged["enable_adaptive"] is True
    assert merged["format"] == "flac"
    assert merged["playlist"] == "/p.m3u"


def test_merge_web_config_checkbox_toggle():
    """Hidden-companion checkboxes toggle off; omitted booleans are preserved."""
    off = merge_web_config(
        {"allow_overwrite": True, "enable_monitoring": True}, {"allow_overwrite": ["0"]}
    )
    assert off["allow_overwrite"] is False
    assert off["enable_monitoring"] is True  # not in this form -> preserved
    on = merge_web_config({"allow_overwrite": False}, {"allow_overwrite": ["0", "on"]})
    assert on["allow_overwrite"] is True


def test_merge_web_config_clears_nullable_text():
    """Empty nullable text clears to None; omitted text is preserved."""
    merged = merge_web_config(
        {"lastfm_api_key": "KEY", "playlist": "/old.m3u", "output": "/music"},
        {"lastfm_api_key": [""], "playlist": [""]},
    )
    assert merged["lastfm_api_key"] is None
    assert merged["playlist"] is None
    assert merged["output"] == "/music"


class FakeProcess:
    def __init__(self, pid: int = 1234, exit_code=None):
        self.pid = pid
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


class ControlledProcess(FakeProcess):
    def __init__(self, pid: int = 1234):
        super().__init__(pid=pid, exit_code=None)
        self.stdin = io.StringIO()
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        self.exit_code = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.exit_code = 143

    def kill(self):
        self.killed = True
        self.exit_code = 137


class HangingProcess(ControlledProcess):
    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("spotify-splitter", timeout)


def write_status(path: Path, **overrides):
    data = {
        "schema_version": 1,
        "pid": 1234,
        "state": "recording",
        "current_track": {
            "artist": "Ada",
            "title": "Status Song",
            "album": "Signals",
            "duration_ms": 200000,
            "position": 1.5,
        },
        "tracks_recorded": 7,
        "timer": {
            "enabled": True,
            "elapsed_seconds": 10,
            "remaining_seconds": 50,
        },
        "audio": {
            "queue_depth": 4,
            "dropped_frames": 2,
            "buffer_warnings": 1,
        },
        "last_error": None,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_command_passes_status_and_history_files(tmp_path):
    from spoti2_service.service_app import RECORDER_HISTORY_PATH

    status_path = tmp_path / "status.json"
    supervisor = RecorderSupervisor(status_path=status_path)

    command = supervisor._build_command(DEFAULT_CONFIG.copy())

    assert command[command.index("--status-file") + 1] == str(status_path)
    assert command[command.index("--history-file") + 1] == str(RECORDER_HISTORY_PATH)
    assert command[-6:] == [
        "record", "--status-file", str(status_path),
        "--control-stdin", "--history-file", str(RECORDER_HISTORY_PATH),
    ]


def test_render_index_escapes_config_values():
    """Server-rendered config values must be escaped into input attributes."""
    config = DEFAULT_CONFIG.copy()
    config["output"] = '"/tmp/music"'
    config["lastfm_api_key"] = '<script>alert("x")</script>'
    status = {"state": "running", "details": "<b>recording</b>"}

    html = render_index(config, status, verbose_logging=True)

    assert "Spytify-LE - Linux Audio Recorder" in html
    # Config values are server-rendered into attributes and must be escaped.
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in html
    assert '&quot;/tmp/music&quot;' in html
    assert 'name="verbose" checked' in html
    # Live state/details are populated client-side via textContent, not server HTML.
    assert "<b>recording</b>" not in html


def test_stop_sends_graceful_control_command(tmp_path):
    supervisor = RecorderSupervisor(
        status_path=tmp_path / "status.json",
        graceful_stop_timeout=0.1,
    )
    process = ControlledProcess()
    supervisor._process = process
    supervisor._set_status("running", "Recording")

    supervisor.stop()

    command = json.loads(process.stdin.getvalue().strip())
    assert command == {"cmd": "stop", "flush": True}
    assert process.terminated is False
    assert process.killed is False
    assert supervisor.status()["state"] == "stopped"


def test_stop_falls_back_to_terminate_after_control_timeout(tmp_path):
    supervisor = RecorderSupervisor(
        status_path=tmp_path / "status.json",
        graceful_stop_timeout=0.1,
    )
    process = HangingProcess()
    supervisor._process = process
    supervisor._set_status("running", "Recording")

    supervisor.stop()

    command = json.loads(process.stdin.getvalue().strip())
    assert command == {"cmd": "stop", "flush": True}
    assert process.terminated is True
    assert process.killed is True
    assert supervisor._process is None
    assert supervisor.status()["state"] == "stopped"


def test_status_without_file_returns_supervisor_state(tmp_path):
    supervisor = RecorderSupervisor(status_path=tmp_path / "missing-status.json")
    supervisor._set_status("starting", "Starting recorder")

    status = supervisor.status()

    assert status["state"] == "starting"
    assert status["details"] == "Starting recorder"
    assert "recorder_status_stale" not in status


def test_status_merges_fresh_recorder_detail(tmp_path):
    status_path = tmp_path / "status.json"
    write_status(status_path)
    supervisor = RecorderSupervisor(status_path=status_path)
    supervisor._process = FakeProcess(pid=1234)
    supervisor._set_status("running", "Recording")

    status = supervisor.status()

    assert status["state"] == "running"
    assert status["recorder_state"] == "recording"
    assert status["details"] == "🎵 Recording: Ada - Status Song"
    assert status["current_track"] == "Ada - Status Song"
    assert status["tracks_recorded"] == 7
    assert status["timer_enabled"] is True
    assert status["timer_remaining_seconds"] == 50
    assert status["queue_depth"] == 4
    assert status["dropped_frames"] == 2
    assert status["recorder_status_stale"] is False


def test_status_ignores_stale_pid(tmp_path):
    status_path = tmp_path / "status.json"
    write_status(status_path, pid=9999)
    supervisor = RecorderSupervisor(status_path=status_path)
    supervisor._process = FakeProcess(pid=1234)
    supervisor._set_status("running", "Recording")

    status = supervisor.status()

    assert status["state"] == "running"
    assert status["details"] == "Recording"
    assert status["current_track"] == ""
    assert status["recorder_status_stale"] is True
    assert "tracks_recorded" not in status


def test_status_process_snapshot_survives_concurrent_clear(tmp_path):
    status_path = tmp_path / "status.json"
    write_status(status_path)
    supervisor = RecorderSupervisor(status_path=status_path)

    class ClearingProcess(FakeProcess):
        def poll(self):
            supervisor._process = None
            return None

    supervisor._process = ClearingProcess(pid=1234)
    supervisor._set_status("running", "Recording")

    status = supervisor.status()

    assert status["current_track"] == "Ada - Status Song"
    assert status["recorder_status_stale"] is False


def test_paused_status_uses_matching_pid_even_when_timestamp_is_old(tmp_path):
    status_path = tmp_path / "status.json"
    old_timestamp = (
        datetime.now(timezone.utc) - timedelta(minutes=5)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    write_status(status_path, updated_at=old_timestamp)
    supervisor = RecorderSupervisor(status_path=status_path)
    supervisor._process = FakeProcess(pid=1234)
    supervisor._pause_event.set()
    supervisor._set_status("paused", "Paused")

    status = supervisor.status()

    assert status["state"] == "paused"
    assert status["details"] == "Paused"
    assert status["current_track"] == "Ada - Status Song"
    assert status["recorder_status_stale"] is False
