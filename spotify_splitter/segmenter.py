from pathlib import Path
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC, ID3
import mutagen
import numpy as np
import requests
import logging
from typing import List, Optional

from .mpris import TrackInfo

def is_song(track: TrackInfo) -> bool:
    """Return ``True`` if ``track`` looks like a real Spotify song.

    Ads usually expose a non standard ``mpris:trackid`` or omit it entirely.
    A genuine track will usually have a URI starting with ``spotify:track:`` or
    a D-Bus object path starting with ``/com/spotify/track/``.
    """
    if not track or not track.id:
        return False
    track_id_str = str(track.id)
    return track_id_str.startswith("spotify:track:") or track_id_str.startswith(
        "/com/spotify/track/"
    )

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
        self.is_first_track = True
        self.current_complete = True

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
        if self.current is not None:
            if self.is_first_track:
                self.is_first_track = False
            else:
                self.flush()
        self.buffer.clear()

        if is_song(track):
            self.current = track
            self.recording = True
            self.current_complete = track.position <= 2_000_000
            logger.info("▶ Recording: %s – %s", track.artist, track.title)
        else:
            # Treat ads as gaps; frames will be ignored until the next track
            self.current = None
            self.current_complete = False
            self.recording = False
            logger.info("⏩ Ad detected, skipping recording")

    def flush(self) -> None:
        if not self.current or not self.buffer:
            return
        if not self.current_complete:
            self.buffer.clear()
            return
        raw = np.concatenate(self.buffer)
        self._export(raw, self.current)
        self.buffer.clear()

    def _export(self, segment: np.ndarray, t: TrackInfo) -> None:
        if segment.dtype.kind == "f":
            segment = np.clip(segment, -1.0, 1.0)
            segment = (
                segment * np.iinfo(np.int16).max
            ).astype(np.int16)
        audio_segment = AudioSegment(
            segment.tobytes(),
            frame_rate=self.samplerate,
            sample_width=2,
            channels=segment.shape[1],
        )
        safe_artist = sanitize(t.artist)
        safe_album = sanitize(t.album)
        safe_title = sanitize(t.title)

        folder = self.output_dir / safe_artist / safe_album
        folder.mkdir(parents=True, exist_ok=True)
        num_prefix = ""
        if t.track_number:
            try:
                num_prefix = f"{int(t.track_number):02d} - "
            except Exception:
                num_prefix = f"{t.track_number} - "
        path = folder / f"{num_prefix}{safe_title}.{self.format}"
        audio_segment.export(path, format=self.format, bitrate="320k")

        try:
            tags = EasyID3(path)
        except Exception:
            tags = EasyID3()

        tags["artist"] = t.artist
        tags["title"] = t.title
        tags["album"] = t.album
        if t.track_number:
            tags["tracknumber"] = str(t.track_number)
        tags.save(path)

        try:
            audio = ID3(path)
        except Exception as e:
            logger.warning("Could not load file for tagging: %s", e)
            return

        if t.art_uri:
            try:
                img = requests.get(t.art_uri).content
                audio.add(APIC(3, "image/jpeg", 3, "Front cover", img))
            except Exception as e:
                logger.warning("Failed to download or embed cover art: %s", e)

        audio.save()
        logger.info("Saved %s", path)
