import sys
import types
import importlib
import numpy as np
import pytest


def load_segmenter(monkeypatch):
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(sys.modules, "gi.repository", types.SimpleNamespace(GLib=types.SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "pydbus", types.ModuleType("pydbus"))
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

    track = TrackInfo("Artist", "Title", "Album", None, "1")
    manager.start_track(track)
    manager.add_frames(np.zeros((2, 2), dtype="float32"))
    manager.flush()

    assert exported and exported[0].title == "Title"


def test_pause_resume(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="mp3")
    track = TrackInfo("Artist", "Title", "Album", None, "1")
    manager.start_track(track)
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.pause_recording()
    manager.add_frames(np.ones((2, 2), dtype="float32"))
    manager.resume_recording()
    manager.add_frames(np.ones((2, 2), dtype="float32"))

    assert len(manager.buffer) == 2
