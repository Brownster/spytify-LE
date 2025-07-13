import sounddevice as sd
import numpy as np
from queue import Queue, Full
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioStream:
    """Continuously capture audio from a PipeWire/PulseAudio monitor."""

    def __init__(self, monitor_name: str, samplerate: int = 44100, channels: int = 2, q: Optional[Queue] = None):
        self.q: Queue[np.ndarray] = q or Queue(maxsize=20)

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
            logger.debug("Exact device name match failed. Searching for a partial match...")
            try:
                devices = sd.query_devices()
            except Exception:  # pragma: no cover - requires PortAudio
                raise

            search_term = monitor_name
            if "alsa_output" in search_term and ".monitor" in search_term:
                # use the descriptive portion of the monitor name which often
                # matches what PortAudio reports
                search_term = search_term.split(".")[1].replace("_", " ")

            for idx, dev in enumerate(devices):
                name = str(dev.get("name", ""))
                if search_term in name:
                    logger.debug(
                        "Resolved monitor %s -> device %s (%s)", monitor_name, idx, name
                    )
                    self.stream = _open(idx)
                    break
            else:
                raise ValueError(f"Could not find a matching sounddevice for '{monitor_name}'")

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
