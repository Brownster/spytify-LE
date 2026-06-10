"""In-place year/genre correction for already-tagged tracks.

LastFM is often wrong (re-release dates, mis-tagged genres). This rewrites the
``date``/``genre`` tags of a saved file via mutagen — no re-encode — after
validating the path is inside the recording output directory. See
``docs/track-history-design.md`` Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mutagen import File as MutagenFile


class MetadataEditError(Exception):
    """Raised when a metadata edit is rejected or fails."""


def validate_year(year: Optional[str]) -> Optional[int]:
    """Return a validated 1900–2100 year int, or None to clear. Raises on invalid."""
    if year is None or str(year).strip() == "":
        return None
    try:
        value = int(str(year).strip())
    except ValueError:
        raise MetadataEditError("Year must be a number")
    if not (1900 <= value <= 2100):
        raise MetadataEditError("Year must be between 1900 and 2100")
    return value


def _safe_target(path: str, output_dir: str) -> Path:
    """Resolve ``path`` and ensure it sits inside ``output_dir`` (no escapes)."""
    if not path:
        raise MetadataEditError("Missing file path")
    target = Path(path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise MetadataEditError("Refusing to edit a file outside the output directory")
    if not target.is_file():
        raise MetadataEditError("File not found")
    return target


def edit_track_metadata(
    path: str,
    output_dir: str,
    year: Optional[int],
    genre: Optional[str],
) -> Path:
    """Rewrite the ``date``/``genre`` tags of ``path`` in place.

    Empty/None ``year``/``genre`` clears that tag. Returns the resolved path.
    Raises :class:`MetadataEditError` on validation or tagging failure.
    """
    target = _safe_target(path, output_dir)
    try:
        audio = MutagenFile(target, easy=True)
    except Exception as e:
        raise MetadataEditError(f"Could not open file for tagging: {e}")
    if audio is None:
        raise MetadataEditError("Unsupported audio file for tagging")

    if audio.tags is None:
        try:
            audio.add_tags()
        except Exception:
            pass

    if year is not None:
        audio["date"] = str(year)
    else:
        audio.pop("date", None)

    if genre:
        audio["genre"] = genre
    else:
        audio.pop("genre", None)

    try:
        audio.save()
    except Exception as e:
        raise MetadataEditError(f"Failed to write tags: {e}")
    return target
