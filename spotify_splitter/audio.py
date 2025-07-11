import sounddevice as sd
import numpy as np
from queue import Queue, Full
import logging

logger = logging.getLogger(__name__)


class AudioStream:
    """Continuously capture audio from a PipeWire/PulseAudio monitor."""

    def __init__(self, monitor_name: str, samplerate: int = 44100, channels: int = 2):
        self.q: Queue[np.ndarray] = Queue(maxsize=20)

        def _open(device):
            return sd.InputStream(
                device=device,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
                callback=self._callback,
            )

        try:
            self.stream = _open(monitor_name)
        except Exception:
            # ``monitor_name`` may not exactly match a PortAudio device.
            try:
                devices = sd.query_devices()
            except Exception:  # pragma: no cover - requires PortAudio
                raise
            for idx, dev in enumerate(devices):
                name = str(dev.get("name", ""))
                if monitor_name in name.replace(" ", "").replace(",", ""):  # simple substring match
                    logger.debug("Resolved monitor %s -> device %s (%s)", monitor_name, idx, name)
                    self.stream = _open(idx)
                    break
            else:
                raise

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
