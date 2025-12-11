"""Duration parsing and formatting utilities for recording timer feature."""

import re
from typing import Tuple


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to total seconds.

    Supported formats:
    - "4h29m" -> 16140 seconds
    - "2h30m" -> 9000 seconds
    - "90m" -> 5400 seconds
    - "5400s" -> 5400 seconds
    - "2h" -> 7200 seconds
    - Mixed: "1h30m45s" -> 5445 seconds

    Args:
        duration_str: Duration string to parse

    Returns:
        Total duration in seconds

    Raises:
        ValueError: If format is invalid or duration is zero/negative
    """
    if not duration_str or not isinstance(duration_str, str):
        raise ValueError("Duration string cannot be empty")

    # Remove whitespace
    duration_str = duration_str.strip()

    # Regex pattern to match hours, minutes, and seconds
    # Pattern: optional digits followed by 'h', 'm', or 's'
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?'
    match = re.fullmatch(pattern, duration_str, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration_str}'. "
            "Expected format: '4h29m', '2h30m', '90m', '5400s', or '1h30m45s'"
        )

    hours, minutes, seconds = match.groups()

    # Check if at least one component was provided
    if not any([hours, minutes, seconds]):
        raise ValueError(
            "Duration must include at least hours, minutes, or seconds. "
            "Examples: '4h', '30m', '90s', '1h30m'"
        )

    # Convert to integers (default to 0 if not provided)
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0

    # Calculate total seconds
    total_seconds = (hours * 3600) + (minutes * 60) + seconds

    if total_seconds <= 0:
        raise ValueError("Duration must be positive (greater than 0)")

    return total_seconds


def format_remaining_time(seconds: int) -> str:
    """
    Format remaining seconds as human-readable string.

    Examples:
    - 3661 -> "1h 1m 1s"
    - 125 -> "2m 5s"
    - 45 -> "45s"
    - 0 -> "0s"

    Args:
        seconds: Number of seconds remaining

    Returns:
        Formatted time string
    """
    if seconds < 0:
        seconds = 0

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if nothing else
        parts.append(f"{secs}s")

    return " ".join(parts)


def validate_duration_format(duration_str: str) -> Tuple[bool, str]:
    """
    Validate duration string format without parsing.

    Args:
        duration_str: Duration string to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message will be empty string
    """
    try:
        parse_duration(duration_str)
        return (True, "")
    except ValueError as e:
        return (False, str(e))
