"""Tests for structured recorder status persistence."""

import json

from spotify_splitter.mpris import TrackInfo
from spotify_splitter.recorder_status import (
    AtomicStatusWriter,
    AudioStatus,
    RecorderStatus,
    TimerStatus,
    TrackStatus,
)


def test_recorder_status_defaults():
    status = RecorderStatus()

    data = status.to_dict()

    assert data["schema_version"] == 1
    assert data["state"] == "stopped"
    assert data["current_track"] is None
    assert data["timer"] == {
        "enabled": False,
        "elapsed_seconds": 0,
        "remaining_seconds": 0,
    }
    assert data["audio"] == {
        "queue_depth": 0,
        "dropped_frames": 0,
        "buffer_warnings": 0,
    }
    assert data["updated_at"].endswith("Z")


def test_track_status_from_track_info():
    track = TrackInfo(
        artist="Ada",
        title="First Pass",
        album="Compiler Songs",
        art_uri=None,
        id="track-1",
        track_number=1,
        position=12.5,
        duration_ms=180000,
    )

    assert TrackStatus.from_track_info(track) == TrackStatus(
        artist="Ada",
        title="First Pass",
        album="Compiler Songs",
        duration_ms=180000,
        position=12.5,
    )


def test_atomic_status_writer_writes_json(tmp_path):
    path = tmp_path / "recorder-status.json"
    writer = AtomicStatusWriter(path)
    status = RecorderStatus(
        state="recording",
        current_track=TrackStatus(artist="Artist", title="Title", album="Album"),
        tracks_recorded=3,
        timer=TimerStatus(enabled=True, elapsed_seconds=10, remaining_seconds=50),
        audio=AudioStatus(queue_depth=4, dropped_frames=0, buffer_warnings=1),
    )

    writer.write(status)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == writer.read()
    assert data["state"] == "recording"
    assert data["current_track"]["title"] == "Title"
    assert data["timer"]["remaining_seconds"] == 50
    assert data["audio"]["queue_depth"] == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_status_writer_replaces_existing_file(tmp_path):
    path = tmp_path / "recorder-status.json"
    writer = AtomicStatusWriter(path)

    writer.write(RecorderStatus(state="starting"))
    writer.write(RecorderStatus(state="stopped", last_error="done"))

    data = writer.read()
    assert data["state"] == "stopped"
    assert data["last_error"] == "done"


def test_atomic_status_writer_can_skip_fsync(tmp_path, monkeypatch):
    path = tmp_path / "recorder-status.json"
    writer = AtomicStatusWriter(path, fsync=False)
    fsync_calls = []
    monkeypatch.setattr("spotify_splitter.recorder_status.os.fsync", fsync_calls.append)

    writer.write(RecorderStatus(state="recording"))

    assert writer.read()["state"] == "recording"
    assert fsync_calls == []
