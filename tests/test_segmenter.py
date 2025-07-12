import sys
import types
import importlib
import numpy as np
import pytest
from pydub import AudioSegment


def load_segmenter(monkeypatch):
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(sys.modules, "gi.repository", types.SimpleNamespace(GLib=types.SimpleNamespace()))
    dummy_dbus = types.ModuleType("pydbus")
    dummy_dbus.SessionBus = lambda: None
    monkeypatch.setitem(sys.modules, "pydbus", dummy_dbus)
    module = importlib.import_module("spotify_splitter.segmenter")
    importlib.reload(module)
    return module


def test_sanitize(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    assert segmenter.sanitize("AC/DC") == "ACDC"


def test_segment_manager_flush(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    exported = []
    monkeypatch.setattr(manager, "_export", lambda seg, t: exported.append(t))

    track = TrackInfo("Artist", "Title", "Album", None, "spotify:track:1", 1, 0)
    manager.start_track(track)
    manager.add_frames(np.zeros((2, 2), dtype="float32"))
    manager.flush()

    assert exported and exported[0].title == "Title"


def test_pause_resume(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    track = TrackInfo("Artist", "Title", "Album", None, "spotify:track:1", 1, 0)
    manager.start_track(track)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.pause_recording()
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.resume_recording()
    manager.add_frames(np.ones((2, 2), dtype="float32"))

    assert len(manager.buffer) == 2


def test_skip_ad(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    exported = []
    monkeypatch.setattr(manager, "_export", lambda seg, t: exported.append(t))

    ad = TrackInfo("AdArtist", "AdTitle", "AdAlbum", None, "spotify:ad:123", None, 0)
    manager.start_track(ad)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.flush()

    assert not exported


def test_only_complete_tracks_saved(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    exported = []
    monkeypatch.setattr(manager, "_export", lambda seg, t: exported.append(t.title))

    t1 = TrackInfo("A1", "T1", "Al1", None, "spotify:track:1", 1, 0)
    t2 = TrackInfo("A2", "T2", "Al2", None, "spotify:track:2", 2, 0)
    t3 = TrackInfo("A3", "T3", "Al3", None, "spotify:track:3", 3, 0)

    manager.start_track(t1)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.start_track(t2)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.start_track(t3)
    manager.add_frames(np.ones((2, 2), dtype="float32"))

    assert exported == ["T2"]


def test_incomplete_track_discarded(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    exported = []
    monkeypatch.setattr(manager, "_export", lambda seg, t: exported.append(t.title))

    t1 = TrackInfo("A1", "T1", "Al1", None, "spotify:track:1", 1, 0)
    t2 = TrackInfo("A2", "T2", "Al2", None, "spotify:track:2", 2, 5_000_000)
    t3 = TrackInfo("A3", "T3", "Al3", None, "spotify:track:3", 3, 0)

    manager.start_track(t1)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.start_track(t2)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.start_track(t3)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.flush()

    assert exported == ["T3"]


def test_is_song_new_format(monkeypatch):
    """Track IDs starting with '/com/spotify/track/' are considered songs."""
    segmenter = load_segmenter(monkeypatch)
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    song = TrackInfo("Artist", "Title", "Album", None, "/com/spotify/track/123", 1, 0)
    assert segmenter.is_song(song)


def test_float_export_not_distorted(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    track = TrackInfo("Artist", "Tone", "Album", None, "spotify:track:1", 1, 0)

    t = np.linspace(0, 1, manager.samplerate, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    frames = np.stack([sine, sine], axis=1).astype("float32")

    manager._export(frames, track)

    exported = tmp_path / "Artist" / "Album" / "01 - Tone.mp3"
    assert exported.exists()

    ref = (np.clip(frames, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    ref_seg = AudioSegment(
        ref.tobytes(),
        frame_rate=manager.samplerate,
        sample_width=2,
        channels=2,
    )
    ref_path = tmp_path / "ref.mp3"
    ref_seg.export(ref_path, format="mp3", bitrate="320k")

    out = AudioSegment.from_file(exported)
    ref_out = AudioSegment.from_file(ref_path)
    out_arr = np.array(out.get_array_of_samples())
    ref_arr = np.array(ref_out.get_array_of_samples())
    diff = np.mean(np.abs(out_arr - ref_arr))
    assert diff < 1
