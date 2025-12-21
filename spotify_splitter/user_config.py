import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_OUTPUT_DIR = Path.home() / "Music" / "Spotify Splitter"
CONFIG_DIR_NAME = "spotify_splitter"
CONFIG_FILENAME = "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "output": str(DEFAULT_OUTPUT_DIR),
    "format": "mp3",
    "player": "spotify",
    "profile": "auto",
    "enable_adaptive": True,
    "enable_monitoring": True,
    "enable_metrics": True,
    "debug_mode": False,
    "playlist": None,
    "bundle_playlist": False,
    "bundle_album_art_uri": None,  # Custom album artwork URL for bundle playlists
    "playlist_base_path": None,  # Base path for M3U file entries (for NAS/remote server mapping)
    "max_duration": None,  # Maximum recording duration (e.g., "4h29m")
    "queue_size": None,
    "blocksize": None,
    "latency": None,
    "max_buffer_size": 1000,
    "min_buffer_size": 50,
    "lastfm_api_key": None,  # LastFM API key for metadata fetching
    "allow_overwrite": False,  # Allow overwriting existing track files
}


def _expand(path: str) -> str:
    """Expand '~' and environment variables in a path string."""
    return os.path.expandvars(os.path.expanduser(path))


def get_config_path(custom_path: Optional[str] = None) -> Path:
    """Return the path to the configuration file."""
    if custom_path:
        return Path(_expand(custom_path))

    base_dir = os.environ.get("XDG_CONFIG_HOME")
    if base_dir:
        base_path = Path(base_dir)
    else:
        base_path = Path.home() / ".config"
    return base_path / CONFIG_DIR_NAME / CONFIG_FILENAME


def load_user_config(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from disk, falling back to defaults if missing."""
    config_path = get_config_path(custom_path)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with config_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (json.JSONDecodeError, OSError):
        # Return defaults if file is unreadable
        return DEFAULT_CONFIG.copy()

    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return merged


def save_user_config(config: Dict[str, Any], custom_path: Optional[str] = None) -> Path:
    """Persist configuration to disk and return the resulting path."""
    config_path = get_config_path(custom_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }

    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(serializable_config, fp, indent=2, sort_keys=True)

    return config_path


def apply_cli_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge CLI overrides into the configuration."""
    result = config.copy()
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result

