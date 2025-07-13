import sys
import types
import importlib
import queue
from pathlib import Path
import numpy as np
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


def test_process_segments(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackMarker = segmenter.TrackMarker
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    audio_q = queue.Queue()
    event_q = queue.Queue()
    manager = SegmentManager(44100, output_dir=tmp_path, fmt="wav", audio_queue=audio_q, event_queue=event_q)

    exported = []
    monkeypatch.setattr(manager, "_export", lambda seg, t: exported.append(t.title))
    monkeypatch.setattr(segmenter, "split_on_silence", lambda window, **kw: [window])

    frames1 = np.zeros((100, 2), dtype="float32")
    frames2 = np.ones((100, 2), dtype="float32")
    audio_q.put(frames1)
    manager._ingest_audio()
    manager.track_markers.append(TrackMarker(len(manager.continuous_buffer), TrackInfo("A", "T1", "Al", None, "spotify:track:1", 1, 0)))
    audio_q.put(frames2)
    manager._ingest_audio()
    manager.track_markers.append(TrackMarker(len(manager.continuous_buffer), TrackInfo("A", "T2", "Al", None, "spotify:track:2", 2, 0)))

    manager.process_segments()
    assert exported == ["T1"]


def test_is_song_new_format(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    song = TrackInfo("Artist", "Title", "Album", None, "/com/spotify/track/123", 1, 0)
    assert segmenter.is_song(song)


def test_float_export_not_distorted(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="wav")
    track = TrackInfo("Artist", "Tone", "Album", None, "spotify:track:1", 1, 0)

    t = np.linspace(0, 1, manager.samplerate, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    frames = np.stack([sine, sine], axis=1).astype("float32")

    captured = {}
    def fake_export(self, path, format=None, bitrate=None):
        captured['segment'] = self
        Path(path).touch()

    monkeypatch.setattr(AudioSegment, 'export', fake_export)

    manager._export(frames, track)

    exported = tmp_path / "Artist" / "Album" / "01 - Tone.wav"
    assert exported.exists()

    out_arr = np.array(captured['segment'].get_array_of_samples()).reshape(-1, 2)
    ref = (np.clip(frames, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    ref_arr = ref.reshape(-1, 2)
    diff = np.mean(np.abs(out_arr - ref_arr))
    assert diff < 1


def test_skip_existing_file(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="wav")
    track = TrackInfo("Artist", "Title", "Album", None, "spotify:track:1", 1, 0)

    existing = manager._get_track_path(track)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.touch()

    called = []
    monkeypatch.setattr(AudioSegment, "export", lambda *a, **k: called.append(True))

    manager._export(np.ones((2, 2), dtype="float32"), track)

    assert not called
