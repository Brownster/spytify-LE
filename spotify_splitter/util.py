import subprocess
import json
import logging

logger = logging.getLogger(__name__)


def find_spotify_monitor() -> str:
    """Return the monitor source name for the active Spotify sink."""
    out = subprocess.check_output(["pactl", "-f", "json", "list", "sink-inputs"]).decode()
    inputs = json.loads(out)
    for inp in inputs:
        if inp.get("properties", {}).get("application.name") == "Spotify":
            sink = inp["sink"]
            sinks = json.loads(subprocess.check_output(["pactl", "-f", "json", "list", "sinks"]))
            for s in sinks:
                if s["index"] == sink:
                    logger.debug("Found Spotify monitor %s", s["monitor_source_name"])
                    return s["monitor_source_name"]
    raise RuntimeError("Spotify sink not found – is music playing?")
