from gi.repository import GLib
from pydbus import SessionBus
from collections import namedtuple
from typing import Callable, Optional
import logging
import json
try:
    from pydbus.errors import DBusError
except Exception:  # pragma: no cover - fallback if gi is missing
    class DBusError(Exception):
        pass

TrackInfo = namedtuple(
    "TrackInfo",
    "artist title album art_uri id track_number position duration_ms",
)

logger = logging.getLogger(__name__)


def track_events(
    on_change: Callable[[TrackInfo], None],
    on_status: Optional[Callable[[str], None]] = None,
    dump_metadata: bool = False,
    player_name: str = "spotify",
) -> None:
    """Subscribe to MPRIS track change events from a Spotify player.

    ``on_change`` is called with a :class:`TrackInfo` whenever the track updates.
    If provided, ``on_status`` is called with the string ``PlaybackStatus`` when
    the player status changes.

    ``dump_metadata`` prints raw metadata dictionaries for debugging.
    ``player_name`` selects the MPRIS service (e.g. ``spotify`` or ``spotifyd``).
    """
    bus = SessionBus()
    service_name = f"org.mpris.MediaPlayer2.{player_name}"
    logger.debug("Connecting to MPRIS service: %s", service_name)
    try:
        spotify = bus.get(service_name, "/org/mpris/MediaPlayer2")
    except DBusError as e:
        logger.error(
            "Could not connect to D-Bus service '%s'. Is the player running?",
            service_name,
        )
        raise e

    def handler(_iface, changed, _invalid):
        md = changed.get("Metadata", {})
        if md:
            if dump_metadata:
                print(json.dumps(md, indent=2))
            try:
                position = spotify.Position
            except Exception:
                position = 0
            length = md.get("mpris:length")
            duration_ms = int(length / 1000) if length else 0
            track = TrackInfo(
                artist=md.get("xesam:artist", ["Unknown"])[0],
                title=md.get("xesam:title", "Unknown"),
                album=md.get("xesam:album", "Unknown"),
                art_uri=md.get("mpris:artUrl"),
                id=md.get("mpris:trackid"),
                track_number=md.get("xesam:trackNumber"),
                position=position,
                duration_ms=duration_ms,
            )
            logger.debug("Track changed: %s - %s", track.artist, track.title)
            on_change(track)

        status = changed.get("PlaybackStatus")
        if status and on_status:
            logger.debug("Playback status changed: %s", status)
            on_status(status)

    # Emit current metadata before listening for changes so the first track
    # is captured even if playback was already running when the program starts.
    try:
        initial = spotify.Metadata
        handler(None, {"Metadata": initial}, None)
    except Exception:
        logger.debug("No initial metadata available")

    if on_status:
        try:
            on_status(spotify.PlaybackStatus)
        except Exception:
            logger.debug("No initial playback status available")

    spotify.onPropertiesChanged = handler
    loop = GLib.MainLoop()
    logger.debug("Entering GLib main loop for MPRIS events")
    loop.run()
