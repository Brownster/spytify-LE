import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def load_segmenter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(
        sys.modules,
        "gi.repository",
        types.SimpleNamespace(GLib=types.SimpleNamespace()),
    )
    dummy_dbus = types.ModuleType("pydbus")
    dummy_dbus.SessionBus = lambda: None
    monkeypatch.setitem(sys.modules, "pydbus", dummy_dbus)
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    return importlib.import_module("spotify_splitter.segmenter")


def test_chunk_ledger_appends_clipped_int16_chunks(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    ledger = segmenter.ChunkLedger(samplerate=44100, channels=2)

    ledger.append_float32(
        np.array(
            [
                [-2.0, -0.5],
                [0.5, 2.0],
            ],
            dtype=np.float32,
        )
    )

    samples = ledger.slice_frames(0, 2)
    expected = (
        np.array(
            [
                [-1.0, -0.5],
                [0.5, 1.0],
            ],
            dtype=np.float32,
        )
        * np.iinfo(np.int16).max
    ).astype(np.int16)

    assert ledger.base_frame == 0
    assert ledger.total_frames == 2
    assert ledger.retained_frames == 2
    np.testing.assert_array_equal(samples, expected)


def test_chunk_ledger_slices_across_chunks(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    ledger = segmenter.ChunkLedger(channels=2)

    first = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    second = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    ledger.append_float32(first)
    ledger.append_float32(second)

    samples = ledger.slice_frames(1, 3)
    expected = (
        np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        * np.iinfo(np.int16).max
    ).astype(np.int16)

    np.testing.assert_array_equal(samples, expected)


def test_chunk_ledger_discards_full_and_partial_chunks(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    ledger = segmenter.ChunkLedger(channels=2)

    ledger.append_float32(np.ones((2, 2), dtype=np.float32) * 0.1)
    ledger.append_float32(np.ones((3, 2), dtype=np.float32) * 0.5)
    ledger.discard_before(3)

    assert ledger.base_frame == 3
    assert ledger.total_frames == 5
    assert ledger.retained_frames == 2
    assert len(ledger.chunks) == 1
    assert ledger.chunks[0].start_frame == 3

    expected = (
        np.ones((2, 2), dtype=np.float32) * 0.5 * np.iinfo(np.int16).max
    ).astype(np.int16)
    np.testing.assert_array_equal(ledger.slice_frames(3, 5), expected)


def test_chunk_ledger_materializes_audio_segment(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    ledger = segmenter.ChunkLedger(samplerate=44100, channels=2)

    ledger.append_float32(np.ones((441, 2), dtype=np.float32) * 0.25)
    audio = ledger.to_audio_segment(0, 441)

    assert len(audio) == 10
    assert audio.frame_rate == 44100
    assert audio.channels == 2
    assert audio.sample_width == 2


def test_chunk_ledger_rejects_invalid_ranges(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    ledger = segmenter.ChunkLedger(channels=2)
    ledger.append_float32(np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="beyond total_frames"):
        ledger.slice_frames(0, 3)
    with pytest.raises(ValueError, match="greater than or equal"):
        ledger.slice_frames(2, 1)
    with pytest.raises(ValueError, match="before base_frame"):
        ledger.discard_before(-1)


def test_chunk_ledger_rejects_wrong_shape(monkeypatch):
    segmenter = load_segmenter(monkeypatch)
    ledger = segmenter.ChunkLedger(channels=2)

    with pytest.raises(ValueError, match="shape"):
        ledger.append_float32(np.zeros((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 2 channels"):
        ledger.append_float32(np.zeros((2, 1), dtype=np.float32))
