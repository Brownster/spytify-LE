"""Unit tests for duration parsing utilities."""

import pytest
from spotify_splitter.duration_parser import (
    parse_duration,
    format_remaining_time,
    validate_duration_format,
)


class TestParseDuration:
    """Test cases for parse_duration function."""

    def test_hours_and_minutes(self):
        """Test parsing hours and minutes format."""
        assert parse_duration("4h29m") == 16140
        assert parse_duration("2h30m") == 9000
        assert parse_duration("1h15m") == 4500

    def test_minutes_only(self):
        """Test parsing minutes-only format."""
        assert parse_duration("90m") == 5400
        assert parse_duration("30m") == 1800
        assert parse_duration("1m") == 60

    def test_seconds_only(self):
        """Test parsing seconds-only format."""
        assert parse_duration("5400s") == 5400
        assert parse_duration("60s") == 60
        assert parse_duration("1s") == 1

    def test_hours_only(self):
        """Test parsing hours-only format."""
        assert parse_duration("2h") == 7200
        assert parse_duration("1h") == 3600
        assert parse_duration("5h") == 18000

    def test_mixed_format(self):
        """Test parsing mixed format with hours, minutes, and seconds."""
        assert parse_duration("1h30m45s") == 5445
        assert parse_duration("2h0m30s") == 7230
        assert parse_duration("0h5m10s") == 310

    def test_hours_and_seconds(self):
        """Test parsing hours and seconds (no minutes)."""
        assert parse_duration("1h30s") == 3630
        assert parse_duration("2h45s") == 7245

    def test_minutes_and_seconds(self):
        """Test parsing minutes and seconds (no hours)."""
        assert parse_duration("5m30s") == 330
        assert parse_duration("1m1s") == 61

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert parse_duration("4H29M") == 16140
        assert parse_duration("2H30M") == 9000
        assert parse_duration("90M") == 5400
        assert parse_duration("5400S") == 5400

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is handled."""
        assert parse_duration("  4h29m  ") == 16140
        assert parse_duration("\t90m\n") == 5400

    def test_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("invalid")
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("4hours")
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("4h 29m")  # Space not allowed
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("4h29")  # Missing unit

    def test_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_duration("")
        # Whitespace-only strings get stripped to empty, then fail the "at least one component" check
        with pytest.raises(ValueError, match="must include at least"):
            parse_duration("   ")

    def test_zero_duration(self):
        """Test that zero duration raises ValueError."""
        # Zero durations fail the "must be positive" check
        with pytest.raises(ValueError, match="must be positive"):
            parse_duration("0h0m0s")
        with pytest.raises(ValueError, match="must be positive"):
            parse_duration("0h")
        with pytest.raises(ValueError, match="must be positive"):
            parse_duration("0m")
        with pytest.raises(ValueError, match="must be positive"):
            parse_duration("0s")

    def test_negative_components(self):
        """Test that negative numbers are not accepted by regex."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("-1h")
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("1h-30m")

    def test_large_values(self):
        """Test parsing large duration values."""
        assert parse_duration("24h") == 86400  # 1 day
        assert parse_duration("999h59m59s") == 3599999  # ~42 days


class TestFormatRemainingTime:
    """Test cases for format_remaining_time function."""

    def test_hours_minutes_seconds(self):
        """Test formatting with hours, minutes, and seconds."""
        assert format_remaining_time(3661) == "1h 1m 1s"
        assert format_remaining_time(7265) == "2h 1m 5s"

    def test_minutes_seconds(self):
        """Test formatting with minutes and seconds."""
        assert format_remaining_time(125) == "2m 5s"
        assert format_remaining_time(90) == "1m 30s"

    def test_seconds_only(self):
        """Test formatting with seconds only."""
        assert format_remaining_time(45) == "45s"
        assert format_remaining_time(1) == "1s"

    def test_zero(self):
        """Test formatting zero seconds."""
        assert format_remaining_time(0) == "0s"

    def test_negative_becomes_zero(self):
        """Test that negative values are treated as zero."""
        assert format_remaining_time(-10) == "0s"
        assert format_remaining_time(-3600) == "0s"

    def test_hours_only(self):
        """Test formatting with exact hours (no minutes or seconds)."""
        assert format_remaining_time(3600) == "1h"
        assert format_remaining_time(7200) == "2h"

    def test_minutes_only(self):
        """Test formatting with exact minutes (no seconds)."""
        assert format_remaining_time(60) == "1m"
        assert format_remaining_time(120) == "2m"

    def test_hours_and_minutes(self):
        """Test formatting with hours and minutes (no seconds)."""
        assert format_remaining_time(3660) == "1h 1m"
        assert format_remaining_time(7320) == "2h 2m"

    def test_hours_and_seconds(self):
        """Test formatting with hours and seconds (no minutes)."""
        assert format_remaining_time(3605) == "1h 5s"

    def test_large_durations(self):
        """Test formatting large durations."""
        assert format_remaining_time(86400) == "24h"  # 1 day
        assert format_remaining_time(90061) == "25h 1m 1s"


class TestValidateDurationFormat:
    """Test cases for validate_duration_format function."""

    def test_valid_formats(self):
        """Test that valid formats return (True, '')."""
        assert validate_duration_format("4h29m") == (True, "")
        assert validate_duration_format("90m") == (True, "")
        assert validate_duration_format("5400s") == (True, "")
        assert validate_duration_format("1h30m45s") == (True, "")

    def test_invalid_formats(self):
        """Test that invalid formats return (False, error_message)."""
        is_valid, error = validate_duration_format("invalid")
        assert is_valid is False
        assert "Invalid duration format" in error

        is_valid, error = validate_duration_format("")
        assert is_valid is False
        assert "cannot be empty" in error

        is_valid, error = validate_duration_format("0h0m0s")
        assert is_valid is False
        assert "must be positive" in error


@pytest.mark.parametrize(
    "duration_str,expected_seconds",
    [
        ("4h29m", 16140),
        ("2h30m", 9000),
        ("90m", 5400),
        ("5400s", 5400),
        ("2h", 7200),
        ("1h30m45s", 5445),
        ("1H30M45S", 5445),  # Case insensitive
        ("30m", 1800),
        ("1h", 3600),
    ],
)
def test_parse_duration_parametrized(duration_str, expected_seconds):
    """Parametrized test for common duration formats."""
    assert parse_duration(duration_str) == expected_seconds
