"""Tests for in-place year/genre metadata correction."""

import pytest
from mutagen import File as MutagenFile
from pydub import AudioSegment

from spotify_splitter.metadata_edit import (
    MetadataEditError,
    edit_track_metadata,
    validate_year,
)


def test_validate_year():
    assert validate_year("1985") == 1985
    assert validate_year("") is None
    assert validate_year(None) is None
    with pytest.raises(MetadataEditError):
        validate_year("nope")
    with pytest.raises(MetadataEditError):
        validate_year("1850")


def _make_mp3(path):
    AudioSegment.silent(duration=300).export(path, format="mp3")


def test_edit_rewrites_year_and_genre(tmp_path):
    out = tmp_path / "music"
    f = out / "Artist" / "Album" / "01 - Song.mp3"
    f.parent.mkdir(parents=True)
    _make_mp3(f)

    edit_track_metadata(str(f), str(out), 1985, "jazz")

    audio = MutagenFile(f, easy=True)
    assert audio["date"] == ["1985"]
    assert audio["genre"] == ["jazz"]


def test_edit_clears_when_empty(tmp_path):
    out = tmp_path / "music"
    f = out / "x.mp3"
    out.mkdir()
    _make_mp3(f)
    edit_track_metadata(str(f), str(out), 1985, "jazz")
    edit_track_metadata(str(f), str(out), None, None)

    audio = MutagenFile(f, easy=True)
    assert "date" not in audio
    assert "genre" not in audio


def test_edit_rejects_path_outside_output_dir(tmp_path):
    out = tmp_path / "music"
    out.mkdir()
    outside = tmp_path / "elsewhere.mp3"
    _make_mp3(outside)
    with pytest.raises(MetadataEditError, match="outside the output directory"):
        edit_track_metadata(str(outside), str(out), 1985, "jazz")


def test_edit_rejects_missing_file(tmp_path):
    out = tmp_path / "music"
    out.mkdir()
    with pytest.raises(MetadataEditError, match="not found"):
        edit_track_metadata(str(out / "ghost.mp3"), str(out), 1985, "jazz")
