import subprocess
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamInfo:
    """Information about the Spotify audio stream."""

    monitor_name: str
    samplerate: int
    channels: int


def _is_spotify(properties: dict) -> bool:
    """Return True if the given properties belong to a Spotify stream."""
    spotify_keys = (
        "application.name",
        "application.icon_name",
        "application.process.binary",
        "pipewire.access.portal.app_id",
        "media.name",
    )
    for key in spotify_keys:
        value = properties.get(key)
        if isinstance(value, str) and "spotify" in value.lower():
            return True
    return False


def get_spotify_stream_info() -> StreamInfo:
    """Return :class:`StreamInfo` for the active Spotify stream."""
    out = subprocess.check_output(
        ["pactl", "-f", "json", "list", "sink-inputs"]
    ).decode()
    inputs = json.loads(out)
    for inp in inputs:
        props = inp.get("properties", {})
        if _is_spotify(props):
            sink = inp["sink"]
            spec = inp.get("sample_spec", inp.get("sample_specification"))
            if isinstance(spec, dict):
                rate = spec.get("rate", 44100)
                channels = spec.get("channels", 2)
            elif isinstance(spec, str):
                m = re.search(r"(\d+)ch (\d+)Hz", spec)
                if m:
                    channels = int(m.group(1))
                    rate = int(m.group(2))
                else:
                    rate = 44100
                    channels = 2
            else:
                rate = 44100
                channels = 2
            sinks = json.loads(
                subprocess.check_output(["pactl", "-f", "json", "list", "sinks"])
            )
            for s in sinks:
                if s["index"] == sink:
                    monitor = s.get("monitor_source_name")
                    if not monitor:
                        name = s.get("name")
                        if name:
                            monitor = f"{name}.monitor"
                        else:
                            continue
                    logger.debug("Found Spotify monitor %s", monitor)
                    return StreamInfo(monitor, rate, channels)
    raise RuntimeError("Spotify sink not found – is music playing?")


def find_spotify_monitor() -> str:
    """Backward-compatible wrapper returning only the monitor name."""
    return get_spotify_stream_info().monitor_name
