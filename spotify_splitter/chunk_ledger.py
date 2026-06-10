"""Frame-addressed raw audio ledger for the capture hot path.

``ChunkLedger`` accumulates incoming PCM as int16 chunks keyed by absolute frame
offset, so completed tracks can be sliced and materialized into an ``AudioSegment``
exactly once (instead of an O(n^2) growing ``AudioSegment``). See
``docs/refactor-roadmap.md`` Pass 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from pydub import AudioSegment


@dataclass
class AudioChunk:
    """A retained block of int16 PCM samples with an absolute frame offset."""

    start_frame: int
    samples: np.ndarray


class ChunkLedger:
    """Frame-addressed raw audio ledger for the capture hot path."""

    def __init__(self, samplerate: int = 44100, channels: int = 2) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.base_frame = 0
        self.total_frames = 0
        self.chunks: List[AudioChunk] = []

    @property
    def retained_frames(self) -> int:
        """Return the number of frames still retained in the ledger."""
        return self.total_frames - self.base_frame

    def append_float32(self, frames: np.ndarray) -> None:
        """Clip float32 samples, convert to int16, and append them as one chunk."""
        if frames.size == 0:
            return
        if frames.ndim != 2:
            raise ValueError("frames must have shape (frame_count, channels)")
        if frames.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {frames.shape[1]}"
            )

        clipped = np.clip(frames, -1.0, 1.0)
        samples = (clipped * np.iinfo(np.int16).max).astype(np.int16)
        self.chunks.append(AudioChunk(self.total_frames, samples))
        self.total_frames += samples.shape[0]

    def slice_frames(self, start_frame: int, end_frame: int) -> np.ndarray:
        """Return a copy of samples in the absolute frame range."""
        self._validate_range(start_frame, end_frame)
        if start_frame == end_frame:
            return np.empty((0, self.channels), dtype=np.int16)

        parts = []
        for chunk in self.chunks:
            chunk_start = chunk.start_frame
            chunk_end = chunk_start + chunk.samples.shape[0]
            overlap_start = max(start_frame, chunk_start)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start >= overlap_end:
                continue
            rel_start = overlap_start - chunk_start
            rel_end = overlap_end - chunk_start
            parts.append(chunk.samples[rel_start:rel_end])

        if not parts:
            return np.empty((0, self.channels), dtype=np.int16)
        return np.concatenate(parts, axis=0).copy()

    def to_audio_segment(self, start_frame: int, end_frame: int) -> AudioSegment:
        """Materialize a frame range as an AudioSegment exactly once."""
        samples = self.slice_frames(start_frame, end_frame)
        return AudioSegment(
            samples.tobytes(),
            frame_rate=self.samplerate,
            sample_width=2,
            channels=self.channels,
        )

    def discard_before(self, frame: int) -> None:
        """Discard retained samples before an absolute frame offset."""
        if frame < self.base_frame:
            raise ValueError("cannot discard before base_frame")
        if frame > self.total_frames:
            raise ValueError("cannot discard beyond total_frames")

        retained = []
        for chunk in self.chunks:
            chunk_start = chunk.start_frame
            chunk_end = chunk_start + chunk.samples.shape[0]
            if chunk_end <= frame:
                continue
            if chunk_start < frame:
                trimmed = chunk.samples[frame - chunk_start :].copy()
                retained.append(AudioChunk(frame, trimmed))
            else:
                retained.append(chunk)

        self.chunks = retained
        self.base_frame = frame

    def _validate_range(self, start_frame: int, end_frame: int) -> None:
        if start_frame < self.base_frame:
            raise ValueError("start_frame is before base_frame")
        if end_frame > self.total_frames:
            raise ValueError("end_frame is beyond total_frames")
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
