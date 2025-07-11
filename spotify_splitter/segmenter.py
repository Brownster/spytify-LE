from pathlib import Path
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC
import numpy as np
import requests
import logging
from typing import List, Optional

from .mpris import TrackInfo

def is_song(track: TrackInfo) -> bool:
    """Return ``True`` if ``track`` looks like a real Spotify song.

    Ads usually expose a non standard ``mpris:trackid`` or omit it entirely.
    A genuine track will use a URI starting with ``spotify:track:``.
    """
    if not track or not track.id:
        return False
    return str(track.id).startswith("spotify:track:")

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / "Music"


def sanitize(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    return "".join(c for c in name if c not in r'/\\:*?"<>|')


class SegmentManager:
    """Buffer audio frames and export them per track."""

    def __init__(self, samplerate: int = 44100, output_dir: Path = OUTPUT_DIR, fmt: str = "mp3"):
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.format = fmt
        self.buffer: List[np.ndarray] = []
        self.current: Optional[TrackInfo] = None
        self.recording = True

    def pause_recording(self) -> None:
        """Stop accepting new frames until resumed."""
        logger.info("⏸️ Recording paused")
        self.recording = False

    def resume_recording(self) -> None:
        """Resume accepting frames."""
        logger.info("▶️ Recording resumed")
        self.recording = True

    def add_frames(self, frames: np.ndarray) -> None:
        """Append audio *frames* to the buffer if a song is active."""
        if self.current is not None and self.recording:
            self.buffer.append(frames)

    def start_track(self, track: TrackInfo) -> None:
        """Begin buffering frames for ``track`` if it is not an advertisement."""
        self.flush()
        self.buffer.clear()

        if is_song(track):
            self.current = track
            self.recording = True
            logger.info("▶ Recording: %s – %s", track.artist, track.title)
        else:
            # Treat ads as gaps; frames will be ignored until the next track
            self.current = None
            self.recording = False
            logger.info("⏩ Ad detected, skipping recording")

    def flush(self) -> None:
        if not self.current or not self.buffer:
            return
        raw = np.concatenate(self.buffer)
        segment = AudioSegment(
            raw.tobytes(),
            frame_rate=self.samplerate,
            sample_width=raw.dtype.itemsize,
            channels=raw.shape[1],
        )
        self._export(segment, self.current)
        self.buffer.clear()

    def _export(self, segment: AudioSegment, t: TrackInfo) -> None:
        safe_artist = sanitize(t.artist)
        safe_album = sanitize(t.album)
        safe_title = sanitize(t.title)

        folder = self.output_dir / safe_artist / safe_album
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{safe_artist} - {safe_title}.{self.format}"
        segment.export(path, format=self.format, bitrate="320k")
        audio = EasyID3(path)
        audio["artist"], audio["title"], audio["album"] = t.artist, t.title, t.album
        if t.art_uri:
            try:
                img = requests.get(t.art_uri).content
                audio.tags.add(APIC(3, "image/jpeg", 3, "Front cover", img))
            except Exception as e:
                logger.warning("Failed to download cover art: %s", e)
        audio.save(v2_version=3)
        logger.info("Saved %s", path)
