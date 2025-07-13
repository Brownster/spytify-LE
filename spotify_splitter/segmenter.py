from __future__ import annotations

from pathlib import Path
from collections import namedtuple
import logging
import queue
from typing import List, Optional, Iterable

import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC, ID3
import requests

from .mpris import TrackInfo


logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / "Music"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sanitize(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    return "".join(c for c in name if c not in r'/\\:*?"<>|')


def is_song(track: TrackInfo) -> bool:
    """Return ``True`` if ``track`` looks like a real Spotify song."""
    if not track or not track.id:
        return False
    track_id_str = str(track.id)
    return track_id_str.startswith("spotify:track:") or track_id_str.startswith(
        "/com/spotify/track/"
    )


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

TrackMarker = namedtuple("TrackMarker", "timestamp track_info")


class SegmentManager:
    """Processes a continuous audio stream to find and export complete tracks."""

    def __init__(
        self,
        samplerate: int = 44100,
        output_dir: Path = OUTPUT_DIR,
        fmt: str = "mp3",
        audio_queue: Optional[queue.Queue] = None,
        event_queue: Optional[queue.Queue] = None,
    ) -> None:
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.format = fmt
        self.audio_queue = audio_queue
        self.event_queue = event_queue

        self.continuous_buffer = AudioSegment.empty()
        self.track_markers: List[TrackMarker] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main processing loop; consumes queues until shutdown."""
        if self.audio_queue is None or self.event_queue is None:
            raise RuntimeError("SegmentManager requires audio and event queues")

        while True:
            try:
                event_type, data = self.event_queue.get(timeout=1.0)
                if event_type == "shutdown":
                    break
                if event_type == "track_change":
                    self._ingest_audio()
                    marker = TrackMarker(len(self.continuous_buffer), data)
                    self.track_markers.append(marker)
                    logger.debug("Marker added for track: %s", data.title)
                    self.process_segments()
            except queue.Empty:
                self._ingest_audio()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest_audio(self) -> None:
        """Pull all frames from ``audio_queue`` into the continuous buffer."""
        if not self.audio_queue or self.audio_queue.empty():
            return

        segment = AudioSegment.empty()
        while not self.audio_queue.empty():
            frames = self.audio_queue.get_nowait()
            int_samples = (frames * np.iinfo(np.int16).max).astype(np.int16)
            seg = AudioSegment(
                int_samples.tobytes(),
                frame_rate=self.samplerate,
                sample_width=2,
                channels=frames.shape[1],
            )
            segment += seg
        self.continuous_buffer += segment

    def process_segments(self) -> None:
        """Process the first complete segment using a tiered strategy."""
        if len(self.track_markers) < 2:
            return

        start_marker = self.track_markers[0]
        end_marker = self.track_markers[1]

        search_window = self.continuous_buffer[start_marker.timestamp : end_marker.timestamp]

        if not is_song(start_marker.track_info):
            logger.debug("Previous item was not a song, skipping segment.")
            self.continuous_buffer = self.continuous_buffer[end_marker.timestamp :]
            self.track_markers.pop(0)
            return

        logger.info("Processing segment for: %s", start_marker.track_info.title)

        expected_duration_ms = getattr(start_marker.track_info, "duration_ms", 0)
        song_chunk: Optional[AudioSegment] = None

        # --- Strategy A: Smart Split ---
        logger.debug("Attempting smart split with duration validation...")
        chunks = split_on_silence(
            search_window, min_silence_len=700, silence_thresh=-40, keep_silence=100
        )
        if chunks:
            best_chunk = max(chunks, key=len)
            duration_ratio = len(best_chunk) / expected_duration_ms if expected_duration_ms > 0 else 0
            if expected_duration_ms == 0 or (0.95 < duration_ratio < 1.15):
                logger.debug("Smart split successful. Chunk duration is valid.")
                song_chunk = best_chunk
            else:
                logger.warning(
                    "Smart split found a chunk of unexpected length (found %dms, expected %dms). Proceeding to next strategy.",
                    len(best_chunk), expected_duration_ms,
                )

        # --- Strategy B: Targeted Silence Search Near End ---
        if song_chunk is None and expected_duration_ms > 0:
            logger.debug("Attempting targeted silence search near expected end...")
            search_zone_start = max(0, expected_duration_ms - 2000)
            search_zone = search_window[search_zone_start:]
            silence = detect_silence(search_zone, min_silence_len=500, silence_thresh=-40)
            if silence:
                cut_point = search_zone_start + silence[0][0]
                song_chunk = search_window[:cut_point]
                logger.debug("Targeted search successful. Found silence at %dms.", cut_point)
            else:
                logger.warning("No silence found in targeted search zone.")

        # --- Strategy C: Failsafe Hard Cut ---
        if song_chunk is None:
            logger.warning(
                "All silence detection failed. Falling back to hard cut at metadata event time."
            )
            song_chunk = search_window

        self._export(song_chunk, start_marker.track_info)

        self.continuous_buffer = self.continuous_buffer[end_marker.timestamp :]
        self.track_markers = [
            marker._replace(timestamp=marker.timestamp - end_marker.timestamp)
            for marker in self.track_markers[1:]
        ]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _get_track_path(self, t: TrackInfo) -> Path:
        safe_artist = sanitize(t.artist)
        safe_album = sanitize(t.album)
        safe_title = sanitize(t.title)

        folder = self.output_dir / safe_artist / safe_album
        num_prefix = ""
        if t.track_number:
            try:
                num_prefix = f"{int(t.track_number):02d} - "
            except Exception:
                num_prefix = f"{t.track_number} - "
        return folder / f"{num_prefix}{safe_title}.{self.format}"

    def _export(self, segment: Iterable | AudioSegment, track_info: TrackInfo) -> None:
        """Export ``segment`` to disk and tag it with metadata."""
        if isinstance(segment, np.ndarray):
            if segment.dtype.kind == "f":
                segment = np.clip(segment, -1.0, 1.0)
                segment = (segment * np.iinfo(np.int16).max).astype(np.int16)
            audio_segment = AudioSegment(
                segment.tobytes(),
                frame_rate=self.samplerate,
                sample_width=2,
                channels=segment.shape[1],
            )
        elif isinstance(segment, AudioSegment):
            audio_segment = segment
        else:
            raise TypeError("segment must be numpy array or AudioSegment")

        path = self._get_track_path(track_info)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            logger.info("File %s already exists, skipping export", path)
            return

        audio_segment.export(path, format=self.format, bitrate="320k")

        try:
            tags = EasyID3(path)
        except Exception:
            tags = EasyID3()

        tags["artist"] = track_info.artist
        tags["title"] = track_info.title
        tags["album"] = track_info.album
        if track_info.track_number:
            tags["tracknumber"] = str(track_info.track_number)
        tags.save(path)

        try:
            audio = ID3(path)
        except Exception as e:
            logger.warning("Could not load file for tagging: %s", e)
            return

        if track_info.art_uri:
            try:
                img = requests.get(track_info.art_uri).content
                audio.add(APIC(3, "image/jpeg", 3, "Front cover", img))
            except Exception as e:
                logger.warning("Failed to download or embed cover art: %s", e)

        audio.save()
        logger.info("Saved %s", path)
