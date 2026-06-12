import subprocess
import json
import logging
import re
from dataclasses import dataclass
try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency may be missing
    sd = None

logger = logging.getLogger(__name__)


@dataclass
class StreamInfo:
    """Information about the Spotify audio stream."""

    monitor_name: str
    samplerate: int
    channels: int


def _parse_spec(spec):
    """Return (rate, channels) from a pactl sample_spec dict or string."""
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
    return rate, channels


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
        if isinstance(value, str):
            # Check for Spotify client
            if "spotify" in value.lower():
                return True
            # Check for spotifyd/librespot
            if "librespot" in value.lower():
                return True
            # Check for spotifyd binary name
            if "spotifyd" in value.lower():
                return True
    return False


def _resolve_stream_info(inp: dict, sinks: list) -> "StreamInfo | None":
    """Build :class:`StreamInfo` from a matched pactl sink-input and sink list."""
    sink = inp.get("sink")
    rate, channels = _parse_spec(
        inp.get("sample_spec", inp.get("sample_specification"))
    )
    monitor = None
    for s in sinks:
        if s.get("index") == sink:
            sink_spec = s.get("sample_spec", s.get("sample_specification"))
            if sink_spec:
                rate, channels = _parse_spec(sink_spec)
            monitor = s.get("monitor_source_name")
            if not monitor:
                name = s.get("name")
                if name:
                    monitor = f"{name}.monitor"
            break

    node_name = inp.get("properties", {}).get("node.name")
    if node_name and sd is not None:
        try:
            sd.check_input_settings(device=node_name)
            logger.debug("Found direct match with node.name: %s", node_name)
            return StreamInfo(node_name, rate, channels)
        except Exception:
            logger.debug(
                "node.name '%s' is not a valid sounddevice, continuing...",
                node_name,
            )

    if monitor:
        logger.debug("Found Spotify monitor %s", monitor)
        return StreamInfo(monitor, rate, channels)
    return None


def _spotify_node_names_via_pipewire() -> set:
    """Return ``node.name`` values for Spotify output streams seen by PipeWire.

    Modern flatpak Spotify connects over the PipeWire *native* protocol. Its
    ``application.*`` identity is then stripped from the PulseAudio-compat
    ``sink-input`` view that ``pactl`` exposes, but ``pw-dump`` still carries it.
    We match those nodes so the caller can correlate them with the sink-input
    list by ``node.name``.
    """
    try:
        out = subprocess.check_output(["pw-dump"]).decode()
        objects = json.loads(out)
    except FileNotFoundError:
        logger.debug("pw-dump not available; skipping PipeWire correlation")
        return set()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("pw-dump query failed: %s", e)
        return set()

    names = set()
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {}
        if "Stream/Output/Audio" not in str(props.get("media.class", "")):
            continue
        if _is_spotify(props):
            node_name = props.get("node.name")
            if node_name:
                names.add(node_name)
    if names:
        logger.debug("PipeWire Spotify output nodes: %s", names)
    return names


def get_spotify_stream_info() -> StreamInfo:
    """Return :class:`StreamInfo` for the active Spotify stream."""
    inputs = json.loads(
        subprocess.check_output(["pactl", "-f", "json", "list", "sink-inputs"]).decode()
    )
    sinks = json.loads(
        subprocess.check_output(["pactl", "-f", "json", "list", "sinks"]).decode()
    )

    # Pass 1: identify Spotify directly from sink-input properties (libpulse
    # clients still carry application.name etc.).
    for inp in inputs:
        if _is_spotify(inp.get("properties", {})):
            info = _resolve_stream_info(inp, sinks)
            if info:
                return info

    # Pass 2: PipeWire-native clients (e.g. recent Spotify flatpak) whose app
    # identity is stripped from the pulse-compat view. Correlate by node.name.
    spotify_nodes = _spotify_node_names_via_pipewire()
    if spotify_nodes:
        for inp in inputs:
            node_name = inp.get("properties", {}).get("node.name")
            if node_name in spotify_nodes:
                logger.info(
                    "Identified Spotify via PipeWire node correlation: %s", node_name
                )
                info = _resolve_stream_info(inp, sinks)
                if info:
                    return info

    raise RuntimeError("Spotify sink not found – is music playing?")


def stream_info_for_source(source_name: str) -> StreamInfo:
    """Build :class:`StreamInfo` for an explicitly chosen capture/monitor source.

    Bypasses auto-detection. The source's sample spec is read from ``pactl`` when
    available, falling back to 44100 Hz / 2 channels.
    """
    rate = channels = None
    try:
        sources = json.loads(
            subprocess.check_output(["pactl", "-f", "json", "list", "sources"]).decode()
        )
        for s in sources:
            if s.get("name") == source_name:
                spec = s.get("sample_spec", s.get("sample_specification"))
                if spec:
                    rate, channels = _parse_spec(spec)
                break
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not query source spec for %s: %s", source_name, e)

    # The name may be a PortAudio device (e.g. a PipeWire node) rather than a
    # PulseAudio source; fall back to the device's native settings.
    if rate is None and sd is not None:
        try:
            dev = sd.query_devices(source_name)
            rate = int(dev["default_samplerate"])
            channels = dev["max_input_channels"] or 2
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not query device settings for %s: %s", source_name, e)

    if rate is None:
        rate, channels = 44100, 2
    logger.info(
        "Using explicit capture source: %s (%d Hz, %d ch)", source_name, rate, channels
    )
    return StreamInfo(source_name, rate, channels)
