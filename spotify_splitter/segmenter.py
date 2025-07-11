from pathlib import Path
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC
import numpy as np
import requests
import logging
from typing import List, Optional

from .mpris import TrackInfo

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / "Music" / "SpotifyRips"


class SegmentManager:
    """Buffer audio frames and export them per track."""

    def __init__(self, samplerate: int = 44100, output_dir: Path = OUTPUT_DIR, fmt: str = "mp3"):
        self.samplerate = samplerate
        self.output_dir = output_dir
        self.format = fmt
        self.buffer: List[np.ndarray] = []
        self.current: Optional[TrackInfo] = None

    def add_frames(self, frames: np.ndarray) -> None:
        if self.current is not None:
            self.buffer.append(frames)

    def start_track(self, track: TrackInfo) -> None:
        self.flush()
        self.current = track
        self.buffer.clear()
        logger.info("▶ %s – %s", track.artist, track.title)

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
        folder = self.output_dir / f"{t.artist} / {t.album}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{t.artist} - {t.title}.{self.format}"
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
