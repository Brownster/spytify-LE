import subprocess
import json
import logging

logger = logging.getLogger(__name__)


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


def find_spotify_monitor() -> str:
    """Return the monitor source name for the active Spotify sink."""
    out = subprocess.check_output(["pactl", "-f", "json", "list", "sink-inputs"]).decode()
    inputs = json.loads(out)
    for inp in inputs:
        props = inp.get("properties", {})
        if _is_spotify(props):
            sink = inp["sink"]
            sinks = json.loads(
                subprocess.check_output(["pactl", "-f", "json", "list", "sinks"])
            )
            for s in sinks:
                if s["index"] == sink:
                    logger.debug("Found Spotify monitor %s", s["monitor_source_name"])
                    return s["monitor_source_name"]
    raise RuntimeError("Spotify sink not found – is music playing?")
