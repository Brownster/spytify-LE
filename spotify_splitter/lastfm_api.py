"""LastFM API integration for fetching track metadata (year and genre)."""

from __future__ import annotations

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

import requests


logger = logging.getLogger(__name__)

# LastFM API public key (read-only, safe to include)
# You can register your own at https://www.last.fm/api/account/create
DEFAULT_API_KEY = "450e8b940228b549b766270f90d67740"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"


@dataclass
class LastFMTrackMetadata:
    """Additional metadata from LastFM."""
    year: Optional[int] = None
    genres: Optional[List[str]] = None
    album_art_url: Optional[str] = None


class LastFMAPI:
    """LastFM API client for fetching track metadata."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LastFM API client.

        Args:
            api_key: LastFM API key. If None, uses DEFAULT_API_KEY.
        """
        self.api_key = api_key or DEFAULT_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Spoti2/1.0 (https://github.com/Brownster/spytify-LE)"
        })

        # Cache to avoid redundant API calls
        self._cache: Dict[str, LastFMTrackMetadata] = {}

    def get_track_metadata(self, artist: str, title: str, album: Optional[str] = None) -> LastFMTrackMetadata:
        """
        Fetch track metadata from LastFM API.

        Args:
            artist: Track artist name
            title: Track title
            album: Album name (optional, helps with accuracy)

        Returns:
            LastFMTrackMetadata with year and genre information
        """
        if not artist or not title:
            logger.debug("Missing artist or title, cannot fetch LastFM metadata")
            return LastFMTrackMetadata()

        # Check cache first
        cache_key = f"{artist}|{title}|{album or ''}"
        if cache_key in self._cache:
            logger.debug("Using cached LastFM metadata for %s - %s", artist, title)
            return self._cache[cache_key]

        try:
            # First, get track info (includes genres as tags and year)
            track_info = self._get_track_info(artist, title)

            # If we didn't get year from track, try getting album info
            if not track_info.year and album:
                album_info = self._get_album_info(artist, album)
                if album_info.year:
                    track_info.year = album_info.year

            # Cache the result
            self._cache[cache_key] = track_info

            return track_info

        except Exception as e:
            logger.warning("Failed to fetch LastFM metadata for %s - %s: %s", artist, title, e)
            return LastFMTrackMetadata()

    def _get_track_info(self, artist: str, title: str) -> LastFMTrackMetadata:
        """
        Get track info from LastFM track.getInfo API.

        Returns year (from album) and top tags (as genres).
        """
        try:
            params = {
                "method": "track.getInfo",
                "api_key": self.api_key,
                "artist": artist,
                "track": title,
                "format": "json",
                "autocorrect": 1,  # Enable autocorrect for better matches
            }

            response = self.session.get(LASTFM_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.debug("LastFM API error for %s - %s: %s", artist, title, data.get("message"))
                return LastFMTrackMetadata()

            track = data.get("track", {})

            # Extract year from album release date
            year = None
            album_data = track.get("album", {})
            if album_data:
                # LastFM doesn't always provide release date in track.getInfo
                # We'll need to call album.getInfo separately
                pass

            # Extract top tags as genres (limit to top 5)
            genres = []
            toptags = track.get("toptags", {}).get("tag", [])
            if isinstance(toptags, list):
                genres = [tag["name"] for tag in toptags[:5]]
            elif isinstance(toptags, dict):
                genres = [toptags["name"]]

            logger.debug("LastFM track info for %s - %s: genres=%s", artist, title, genres)

            return LastFMTrackMetadata(year=year, genres=genres if genres else None)

        except requests.RequestException as e:
            logger.debug("LastFM API request failed: %s", e)
            return LastFMTrackMetadata()

    def _get_album_info(self, artist: str, album: str) -> LastFMTrackMetadata:
        """
        Get album info from LastFM album.getInfo API.

        Returns release year from album.
        """
        try:
            params = {
                "method": "album.getInfo",
                "api_key": self.api_key,
                "artist": artist,
                "album": album,
                "format": "json",
                "autocorrect": 1,
            }

            response = self.session.get(LASTFM_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.debug("LastFM album API error for %s - %s: %s", artist, album, data.get("message"))
                return LastFMTrackMetadata()

            album_data = data.get("album", {})

            # Extract year from wiki or tags
            year = None

            # Try to get year from wiki published date
            wiki = album_data.get("wiki", {})
            if wiki and "published" in wiki:
                published = wiki["published"]
                # Published format is usually like "01 Jan 2020, 00:00"
                try:
                    # Extract year from date string
                    parts = published.split(",")
                    if len(parts) >= 1:
                        date_parts = parts[0].split()
                        if len(date_parts) >= 3:
                            year = int(date_parts[2])
                except (ValueError, IndexError):
                    pass

            # Try tags as genres (optional)
            genres = []
            tags = album_data.get("tags", {}).get("tag", [])
            if isinstance(tags, list):
                genres = [tag["name"] for tag in tags[:5]]
            elif isinstance(tags, dict):
                genres = [tags["name"]]

            logger.debug("LastFM album info for %s - %s: year=%s, genres=%s", artist, album, year, genres)

            return LastFMTrackMetadata(year=year, genres=genres if genres else None)

        except requests.RequestException as e:
            logger.debug("LastFM album API request failed: %s", e)
            return LastFMTrackMetadata()

    def clear_cache(self):
        """Clear the metadata cache."""
        self._cache.clear()
        logger.debug("LastFM metadata cache cleared")


# Global instance (can be configured with custom API key)
_global_lastfm_client: Optional[LastFMAPI] = None


def get_lastfm_client(api_key: Optional[str] = None) -> LastFMAPI:
    """
    Get or create the global LastFM API client.

    Args:
        api_key: Optional API key to use. If provided, creates new client.

    Returns:
        LastFM API client instance
    """
    global _global_lastfm_client

    if api_key or _global_lastfm_client is None:
        _global_lastfm_client = LastFMAPI(api_key=api_key)

    return _global_lastfm_client
