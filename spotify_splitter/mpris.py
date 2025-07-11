from gi.repository import GLib
from pydbus import SessionBus
from collections import namedtuple
from typing import Callable, Optional
import logging

TrackInfo = namedtuple("TrackInfo", "artist title album art_uri id track_number")

logger = logging.getLogger(__name__)


def track_events(
    on_change: Callable[[TrackInfo], None],
    on_status: Optional[Callable[[str], None]] = None,
) -> None:
    """Subscribe to MPRIS track change events from Spotify.

    ``on_change`` is called with a :class:`TrackInfo` whenever the track updates.
    If provided, ``on_status`` is called with the string ``PlaybackStatus`` when
    the player status changes.
    """
    bus = SessionBus()
    spotify = bus.get("org.mpris.MediaPlayer2.spotify", "/org/mpris/MediaPlayer2")

    def handler(_iface, changed, _invalid):
        md = changed.get("Metadata", {})
        if md:
            track = TrackInfo(
                artist=md.get("xesam:artist", ["Unknown"])[0],
                title=md.get("xesam:title", "Unknown"),
                album=md.get("xesam:album", "Unknown"),
                art_uri=md.get("mpris:artUrl"),
                id=md.get("mpris:trackid"),
                track_number=md.get("xesam:trackNumber"),
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
