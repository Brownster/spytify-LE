"""Tests for the capped JSONL track history writer."""

import json

from spotify_splitter.track_history import TrackHistoryWriter, TrackResult, SAVED


def test_track_result_to_dict_adds_ts_and_schema():
    data = TrackResult(outcome=SAVED, artist="A", title="T").to_dict()
    assert data["outcome"] == "saved"
    assert data["schema_version"] == 1
    assert data["ts"].endswith("Z")


def test_append_and_read_newest_first(tmp_path):
    writer = TrackHistoryWriter(tmp_path / "history.jsonl")
    writer.append(TrackResult(outcome=SAVED, title="First", year=1972, genre="rock"))
    writer.append(TrackResult(outcome=SAVED, title="Second"))

    records = writer.read()
    assert [r["title"] for r in records] == ["Second", "First"]
    assert records[1]["year"] == 1972
    assert records[1]["genre"] == "rock"
    # File is valid JSONL.
    lines = (tmp_path / "history.jsonl").read_text().splitlines()
    assert all(json.loads(ln) for ln in lines)


def test_history_is_capped(tmp_path):
    writer = TrackHistoryWriter(tmp_path / "history.jsonl", cap=3)
    for i in range(6):
        writer.append(TrackResult(outcome=SAVED, title=f"t{i}"))

    records = writer.read()
    assert len(records) == 3
    assert [r["title"] for r in records] == ["t5", "t4", "t3"]


def test_read_limit_and_missing_file(tmp_path):
    writer = TrackHistoryWriter(tmp_path / "missing.jsonl")
    assert writer.read() == []  # missing file -> empty
    for i in range(5):
        writer.append(TrackResult(outcome=SAVED, title=f"t{i}"))
    assert len(writer.read(limit=2)) == 2
