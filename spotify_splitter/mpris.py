try:
    from gi.repository import GLib
except Exception:  # pragma: no cover - allow running without gi
    GLib = None
try:
    from pydbus import SessionBus
except Exception:  # pragma: no cover - allow running without pydbus
    SessionBus = None
from collections import namedtuple
from typing import Callable, Optional
import logging
import json
import time
import threading
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


def _poll_for_changes(spotify, handler, on_status, dump_metadata, poll_interval=1.0):
    """Poll MPRIS interface for changes - used for spotifyd compatibility."""
    last_track_id = None
    last_status = None
    
    logger.info("Starting MPRIS polling (interval: %.1fs)", poll_interval)
    
    try:
        while True:
            try:
                # Check for track changes
                current_metadata = spotify.Metadata
                current_track_id = current_metadata.get("mpris:trackid")
                
                if current_track_id != last_track_id:
                    logger.info("Track change detected via polling: %s", current_metadata.get("xesam:title", "Unknown"))
                    if dump_metadata:
                        print(json.dumps(current_metadata, indent=2))
                    handler(None, {"Metadata": current_metadata}, None)
                    last_track_id = current_track_id
                
                # Check for status changes
                if on_status:
                    current_status = spotify.PlaybackStatus
                    if current_status != last_status:
                        logger.debug("Status change detected via polling: %s", current_status)
                        on_status(current_status)
                        last_status = current_status
                        
            except Exception as e:
                logger.error("Error polling MPRIS: %s", e)
                
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        logger.debug("MPRIS polling interrupted")


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
    
    For spotifyd, automatically falls back to polling-based detection due to
    MPRIS signal emission issues.
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
        logger.debug(f"MPRIS properties changed: {changed}")
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

    # Use polling for spotifyd, signals for regular spotify
    if player_name == "spotifyd":
        logger.info("Using polling-based track detection for spotifyd")
        _poll_for_changes(spotify, handler, on_status, dump_metadata)
    else:
        logger.info("Using signal-based track detection for %s", player_name)
        spotify.onPropertiesChanged = handler
        loop = GLib.MainLoop()
        logger.debug("Entering GLib main loop for MPRIS events")
        try:
            loop.run()
        except KeyboardInterrupt:
            logger.debug("MPRIS event loop interrupted")
            loop.quit()
