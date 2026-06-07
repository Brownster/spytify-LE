"""Tests for the web service recorder supervisor."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from spoti2_service.service_app import RecorderSupervisor
from spotify_splitter.user_config import DEFAULT_CONFIG


class FakeProcess:
    def __init__(self, pid: int = 1234, exit_code=None):
        self.pid = pid
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


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


def test_build_command_passes_status_file(tmp_path):
    status_path = tmp_path / "status.json"
    supervisor = RecorderSupervisor(status_path=status_path)

    command = supervisor._build_command(DEFAULT_CONFIG.copy())

    assert "--status-file" in command
    assert command[command.index("--status-file") + 1] == str(status_path)
    assert command[-3:] == ["record", "--status-file", str(status_path)]


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
