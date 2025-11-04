import sys
import types
import importlib
import queue
from pathlib import Path
import numpy as np
from pydub import AudioSegment


def load_segmenter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(sys.modules, "gi.repository", types.SimpleNamespace(GLib=types.SimpleNamespace()))
    dummy_dbus = types.ModuleType("pydbus")
    dummy_dbus.SessionBus = lambda: None
    monkeypatch.setitem(sys.modules, "pydbus", dummy_dbus)
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
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

    frames1 = np.zeros((100, 2), dtype="float32")
    frames2 = np.ones((100, 2), dtype="float32")
    audio_q.put(frames1)
    manager._ingest_audio()
    manager.track_markers.append(TrackMarker(len(manager.continuous_buffer), TrackInfo("A", "T1", "Al", None, "spotify:track:1", 1, 0, 0, None, None)))
    audio_q.put(frames2)
    manager._ingest_audio()
    manager.track_markers.append(TrackMarker(len(manager.continuous_buffer), TrackInfo("A", "T2", "Al", None, "spotify:track:2", 2, 0, 0, None, None)))

    manager.process_segments()
    assert exported == ["T1"]


def test_is_song_new_format(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo
    song = TrackInfo("Artist", "Title", "Album", None, "/com/spotify/track/123", 1, 0, 0, None, None)
    assert segmenter.is_song(song)


def test_float_export_not_distorted(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="wav")
    track = TrackInfo("Artist", "Tone", "Album", None, "spotify:track:1", 1, 0, 0, None, None)

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
    track = TrackInfo("Artist", "Title", "Album", None, "spotify:track:1", 1, 0, 0, None, None)

    existing = manager._get_track_path(track)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.touch()

    called = []
    monkeypatch.setattr(AudioSegment, "export", lambda *a, **k: called.append(True))

    manager._export(np.ones((2, 2), dtype="float32"), track)

    assert not called


def test_playlist_creation(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    playlist = tmp_path / "session.m3u"
    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="wav", playlist_path=playlist)
    track = TrackInfo("Artist", "Title", "Album", None, "spotify:track:1", 1, 0, 0, None, None)

    monkeypatch.setattr(AudioSegment, "export", lambda self, path, format=None, bitrate=None: Path(path).touch())

    manager._export(np.ones((2, 2), dtype="float32"), track)
    manager.close_playlist()

    playlist_content = playlist.read_text().strip().splitlines()
    expected = str(manager._get_track_path(track))
    assert expected in playlist_content[-1]


def test_playlist_append(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    playlist = tmp_path / "session.m3u"
    playlist.parent.mkdir(parents=True, exist_ok=True)
    playlist.write_text("#EXTM3U\nexisting.mp3\n")

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="wav", playlist_path=playlist)
    track = TrackInfo("Artist", "Title", "Album", None, "spotify:track:1", 1, 0, 0, None, None)

    monkeypatch.setattr(AudioSegment, "export", lambda self, path, format=None, bitrate=None: Path(path).touch())

    manager._export(np.ones((2, 2), dtype="float32"), track)
    manager.close_playlist()

    playlist_content = playlist.read_text().strip().splitlines()
    expected = str(manager._get_track_path(track))
    assert playlist_content[0] == "#EXTM3U"
    assert "existing.mp3" in playlist_content
    assert expected in playlist_content[-1]


def test_bundle_playlist_tags(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    playlist = tmp_path / "rain.m3u"
    manager = SegmentManager(
        samplerate=44100,
        output_dir=tmp_path,
        fmt="wav",
        playlist_path=playlist,
        bundle_playlist=True,
    )
    track = TrackInfo("Artist", "Title", "Orig", None, "spotify:track:1", 1, 0, 0, None, None)

    captured = {}

    class FakeEasyID3(dict):
        def __init__(self, path=None):
            pass

        def save(self, path):
            captured.update(self)

    monkeypatch.setattr(segmenter, "EasyID3", FakeEasyID3)
    monkeypatch.setattr(
        segmenter,
        "ID3",
        lambda path: types.SimpleNamespace(add=lambda *a, **k: None, save=lambda: None),
    )
    monkeypatch.setattr(
        AudioSegment,
        "export",
        lambda self, path, format=None, bitrate=None: Path(path).touch(),
    )

    manager._export(np.ones((2, 2), dtype="float32"), track)

    expected_path = tmp_path / "Various Artists" / "rain" / "01 - Title.wav"
    assert expected_path.exists()
    assert captured["album"] == "rain"
    assert captured["albumartist"] == "Various Artists"
    assert captured["artist"] == "Artist"
    assert captured["title"] == "Title"


def test_bundle_playlist_track_numbers(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    playlist = tmp_path / "rain.m3u"
    manager = SegmentManager(
        samplerate=44100,
        output_dir=tmp_path,
        fmt="wav",
        playlist_path=playlist,
        bundle_playlist=True,
    )

    captures = []

    class FakeEasyID3(dict):
        def __init__(self, path=None):
            pass

        def save(self, path):
            captures.append(dict(self))

    monkeypatch.setattr(segmenter, "EasyID3", FakeEasyID3)
    monkeypatch.setattr(
        segmenter,
        "ID3",
        lambda path: types.SimpleNamespace(add=lambda *a, **k: None, save=lambda: None),
    )
    monkeypatch.setattr(
        AudioSegment,
        "export",
        lambda self, path, format=None, bitrate=None: Path(path).touch(),
    )

    track1 = TrackInfo("Artist1", "T1", "Orig1", None, "spotify:track:1", 10, 0, 0, None, None)
    track2 = TrackInfo("Artist2", "T2", "Orig2", None, "spotify:track:2", 20, 0, 0, None, None)

    manager._export(np.ones((2, 2), dtype="float32"), track1)
    manager._export(np.ones((2, 2), dtype="float32"), track2)

    expected1 = tmp_path / "Various Artists" / "rain" / "01 - T1.wav"
    expected2 = tmp_path / "Various Artists" / "rain" / "02 - T2.wav"

    assert expected1.exists()
    assert expected2.exists()
    assert captures[0]["tracknumber"] == "1"
    assert captures[1]["tracknumber"] == "2"
