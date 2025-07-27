import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:5000"


def tag_output(output_dir: Path, playlist_path: Optional[Path] = None, base_url: str = DEFAULT_BASE_URL) -> None:
    """Send MP3 files or playlist to the ID3 tagging API."""
    try:
        if playlist_path:
            payload = {"m3u_path": str(playlist_path)}
            url = f"{base_url}/process_playlist"
            timeout = 600
        else:
            payload = {"directory": str(output_dir)}
            url = f"{base_url}/process_directory"
            timeout = 300

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            logger.info("Tagger processed %s files", result.get("processed", 0))
        else:
            try:
                error = response.json().get("error")
            except Exception:
                error = response.text
            logger.warning("Tagger API error: %s", error)
    except Exception as e:  # pragma: no cover - network or other issues
        logger.warning("Tagger API request failed: %s", e)
