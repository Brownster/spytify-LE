import sounddevice as sd
import numpy as np
from queue import Queue, Full
import logging

logger = logging.getLogger(__name__)


class AudioStream:
    """Continuously capture audio from a PipeWire/PulseAudio monitor."""

    def __init__(self, monitor_name: str, samplerate: int = 44100, channels: int = 2):
        self.q: Queue[np.ndarray] = Queue(maxsize=20)
        self.stream = sd.InputStream(
            device=monitor_name,
            channels=channels,
            samplerate=samplerate,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning("SoundDevice status: %s", status)
        try:
            self.q.put_nowait(indata.copy())
        except Full:
            logger.warning("Audio buffer full; dropping frames")

    def __enter__(self):
        self.stream.start()
        logger.debug("Audio stream started")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stream.stop()
        logger.debug("Audio stream stopped")

    def read(self, timeout: float = 1.0) -> np.ndarray:
        """Blocks until frames are available."""
        return self.q.get(timeout=timeout)
