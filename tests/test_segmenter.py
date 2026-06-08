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
    return importlib.import_module("spotify_splitter.segmenter")


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


def test_ingest_audio_mirrors_frames_to_chunk_ledger(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager

    audio_q = queue.Queue()
    manager = SegmentManager(44100, output_dir=tmp_path, fmt="wav", audio_queue=audio_q)

    frames1 = np.ones((441, 2), dtype="float32") * 0.25
    frames2 = np.ones((441, 2), dtype="float32") * -0.5
    audio_q.put(frames1)
    audio_q.put(frames2)

    manager._ingest_audio()

    assert len(manager.continuous_buffer) == 0
    assert manager.chunk_ledger.total_frames == 882
    assert manager.chunk_ledger.retained_frames == 882
    assert manager._current_buffer_ms() == 20

    ledger_audio = manager.chunk_ledger.to_audio_segment(0, 882)
    ledger_samples = np.array(ledger_audio.get_array_of_samples())
    expected = np.concatenate(
        [
            (np.clip(frames1, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16),
            (np.clip(frames2, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(ledger_samples.reshape((-1, 2)), expected)


def test_continuous_buffer_origin_tracks_clear_and_drop(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager

    audio_q = queue.Queue()
    manager = SegmentManager(44100, output_dir=tmp_path, fmt="wav", audio_queue=audio_q)

    audio_q.put(np.ones((882, 2), dtype="float32") * 0.25)
    manager._ingest_audio()
    manager._clear_continuous_buffer()

    assert len(manager.continuous_buffer) == 0
    assert manager.continuous_buffer_start_frame == 882
    assert manager.chunk_ledger.base_frame == 882
    assert manager.chunk_ledger.retained_frames == 0
    assert manager._buffer_ms_to_frame(0) == 882

    audio_q.put(np.ones((882, 2), dtype="float32") * 0.5)
    manager._ingest_audio()
    manager._drop_continuous_buffer_before(10)

    assert len(manager.continuous_buffer) == 0
    assert manager.continuous_buffer_start_frame == 1323
    assert manager.chunk_ledger.base_frame == 1323
    assert manager.chunk_ledger.retained_frames == 441
    assert manager._buffer_ms_to_frame(10) == 1764


def test_ledger_mirror_channel_mismatch_does_not_break_ingest(monkeypatch, tmp_path, caplog):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager

    audio_q = queue.Queue()
    manager = SegmentManager(44100, output_dir=tmp_path, fmt="wav", audio_queue=audio_q)

    audio_q.put(np.ones((441, 2), dtype="float32") * 0.25)
    manager._ingest_audio()
    audio_q.put(np.ones((441, 1), dtype="float32") * 0.5)

    manager._ingest_audio()

    assert len(manager.continuous_buffer) == 0
    assert manager.chunk_ledger.total_frames == 441
    assert manager.chunk_ledger.retained_frames == 441
    assert "channel mismatch" in caplog.text


def test_frame_markers_bridge_to_boundary_detector_ms(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackMarker = segmenter.TrackMarker
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    audio_q = queue.Queue()
    manager = SegmentManager(44100, output_dir=tmp_path, fmt="wav", audio_queue=audio_q)
    track = TrackInfo("A", "T1", "Al", None, "spotify:track:1", 1, 0, 0, None, None)

    audio_q.put(np.ones((882, 2), dtype="float32") * 0.1)
    manager._ingest_audio()
    manager._clear_continuous_buffer()
    audio_q.put(np.ones((882, 2), dtype="float32") * 0.2)
    manager._ingest_audio()

    manager.track_markers = [
        TrackMarker(0, track, 882),
        TrackMarker(10, track, 1323),
    ]
    exported = []
    captured_markers = []

    def detect_boundary(audio, markers):
        captured_markers.extend(markers)
        return None

    monkeypatch.setattr(manager.boundary_detector, "detect_boundary", detect_boundary)
    monkeypatch.setattr(manager, "_export", lambda seg, t: exported.append(len(seg)))

    manager.process_segments()

    assert [marker.timestamp for marker in captured_markers] == [0, 10]
    assert exported == [10]
    assert manager.continuous_buffer_start_frame == 882
    assert manager.chunk_ledger.base_frame == 882
    assert manager.track_markers[0].timestamp == 10
    assert manager.track_markers[0].frame == 1323


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


def test_artwork_download_uses_manager_session(monkeypatch, tmp_path):
    segmenter = load_segmenter(monkeypatch)
    SegmentManager = segmenter.SegmentManager
    TrackInfo = importlib.import_module("spotify_splitter.mpris").TrackInfo

    manager = SegmentManager(samplerate=44100, output_dir=tmp_path, fmt="wav")
    track = TrackInfo(
        "Artist",
        "Title",
        "Album",
        "https://example.com/art.jpg",
        "spotify:track:1",
        1,
        0,
        0,
        None,
        None,
    )

    calls = []

    class FakeEasyID3(dict):
        def __init__(self, path=None):
            pass

        def save(self, path):
            pass

    class FakeID3:
        def add(self, *args, **kwargs):
            pass

        def save(self):
            pass

    monkeypatch.setattr(segmenter, "EasyID3", FakeEasyID3)
    monkeypatch.setattr(segmenter, "ID3", lambda path: FakeID3())
    monkeypatch.setattr(
        AudioSegment,
        "export",
        lambda self, path, format=None, bitrate=None: Path(path).touch(),
    )
    monkeypatch.setattr(
        manager.artwork_session,
        "get",
        lambda url, timeout=None: calls.append((url, timeout))
        or types.SimpleNamespace(content=b"image-data"),
    )

    manager._export(np.ones((2, 2), dtype="float32"), track)

    assert calls == [("https://example.com/art.jpg", 10)]
