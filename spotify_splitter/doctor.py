"""First-run environment checks for Spotify Splitter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency should be installed
    psutil = None

from .user_config import DEFAULT_CONFIG, get_config_path, load_user_config
from .util import StreamInfo, get_spotify_stream_info


Status = str


@dataclass
class DoctorCheck:
    """One human-actionable environment check."""

    id: str
    label: str
    status: Status
    message: str
    action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    """Complete doctor report for CLI and web UI consumers."""

    checks: List[DoctorCheck]

    @property
    def ok(self) -> bool:
        return not any(check.status == "error" for check in self.checks)

    @property
    def summary(self) -> str:
        errors = sum(1 for check in self.checks if check.status == "error")
        warnings = sum(1 for check in self.checks if check.status == "warning")
        if errors:
            return f"{errors} issue(s) need attention"
        if warnings:
            return f"{warnings} warning(s)"
        return "Ready to record"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
        }


def _run_command(command: List[str], timeout: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _has_spotify_process(process_iter: Optional[Callable[[], Iterable[Any]]] = None) -> bool:
    """Return True when a Spotify/librespot process appears to be running."""
    if process_iter is None:
        if psutil is None:
            return False
        process_iter = lambda: psutil.process_iter(["name", "cmdline"])

    for proc in process_iter():
        try:
            info = getattr(proc, "info", {}) or {}
            name = str(info.get("name") or "")
            cmdline = " ".join(str(part) for part in (info.get("cmdline") or []))
        except Exception:
            continue
        haystack = f"{name} {cmdline}".lower()
        if any(token in haystack for token in ("spotify", "librespot", "spotifyd")):
            return True
    return False


def _check_config(config_path: Optional[str]) -> DoctorCheck:
    path = get_config_path(config_path)
    if not path.exists():
        return DoctorCheck(
            "config",
            "Configuration",
            "warning",
            f"Using defaults; no config file found at {path}",
            "Run spotify-splitter configure or open spotify-splitter web to save settings.",
        )
    try:
        with path.open("r", encoding="utf-8") as fp:
            json.load(fp)
    except Exception as e:
        return DoctorCheck(
            "config",
            "Configuration",
            "error",
            f"Config file is not valid JSON: {e}",
            f"Fix or remove {path}.",
        )
    return DoctorCheck("config", "Configuration", "ok", f"Loaded {path}")


def _check_output_dir(config: Dict[str, Any]) -> DoctorCheck:
    output = Path(str(config.get("output") or DEFAULT_CONFIG["output"])).expanduser()
    if output.exists():
        if output.is_dir() and os_access_writable(output):
            return DoctorCheck("output", "Output folder", "ok", f"Writable: {output}")
        return DoctorCheck(
            "output",
            "Output folder",
            "error",
            f"Output path is not writable: {output}",
            "Choose a writable folder in Settings or with --output.",
        )

    parent = output.parent
    if parent.exists() and os_access_writable(parent):
        return DoctorCheck(
            "output",
            "Output folder",
            "warning",
            f"Folder will be created: {output}",
        )
    return DoctorCheck(
        "output",
        "Output folder",
        "error",
        f"Cannot create output folder under {parent}",
        "Choose a writable output folder.",
    )


def os_access_writable(path: Path) -> bool:
    """Small wrapper to keep tests from patching os.access globally."""
    import os

    return os.access(path, os.W_OK)


def _check_ffmpeg() -> DoctorCheck:
    path = shutil.which("ffmpeg")
    if path:
        return DoctorCheck("ffmpeg", "ffmpeg", "ok", f"Found {path}")
    return DoctorCheck(
        "ffmpeg",
        "ffmpeg",
        "error",
        "ffmpeg is missing",
        "Install ffmpeg, for example: sudo apt install ffmpeg",
    )


def _check_pactl(
    run_command: Callable[[List[str], float], subprocess.CompletedProcess[str]] = _run_command,
) -> DoctorCheck:
    if not shutil.which("pactl"):
        return DoctorCheck(
            "audio_server",
            "PulseAudio/PipeWire",
            "error",
            "pactl is missing",
            "Install PulseAudio/PipeWire tools, for example: sudo apt install pulseaudio-utils",
        )
    try:
        result = run_command(["pactl", "info"], 3.0)
    except Exception as e:
        return DoctorCheck(
            "audio_server",
            "PulseAudio/PipeWire",
            "error",
            f"Could not query audio server: {e}",
            "Start PipeWire/PulseAudio and try again.",
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "pactl info failed").strip()
        return DoctorCheck(
            "audio_server",
            "PulseAudio/PipeWire",
            "error",
            detail,
            "Start PipeWire/PulseAudio and try again.",
        )
    server = ""
    for line in result.stdout.splitlines():
        if line.lower().startswith("server name:"):
            server = line.split(":", 1)[1].strip()
            break
    return DoctorCheck(
        "audio_server",
        "PulseAudio/PipeWire",
        "ok",
        server or "Audio server is reachable",
    )


def _check_pipewire_tools() -> DoctorCheck:
    if shutil.which("pw-dump"):
        return DoctorCheck("pipewire_tools", "PipeWire tools", "ok", "pw-dump is available")
    return DoctorCheck(
        "pipewire_tools",
        "PipeWire tools",
        "warning",
        "pw-dump is missing; Spotify Flatpak detection may be less reliable",
        "Install PipeWire tools if Spotify is installed as a Flatpak.",
    )


def _check_spotify_process(
    process_iter: Optional[Callable[[], Iterable[Any]]] = None,
) -> DoctorCheck:
    if _has_spotify_process(process_iter):
        return DoctorCheck("spotify_process", "Spotify app", "ok", "Spotify is running")
    return DoctorCheck(
        "spotify_process",
        "Spotify app",
        "error",
        "Spotify not detected",
        "Open Spotify and start playback before recording.",
    )


def _check_spotify_stream(
    spotify_running: bool,
    stream_probe: Callable[[], StreamInfo] = get_spotify_stream_info,
) -> DoctorCheck:
    try:
        info = stream_probe()
    except Exception as e:
        if spotify_running:
            return DoctorCheck(
                "spotify_stream",
                "Spotify audio stream",
                "error",
                "Spotify detected but not playing",
                "Press play in Spotify, then run checks again.",
            )
        return DoctorCheck(
            "spotify_stream",
            "Spotify audio stream",
            "error",
            f"No Spotify stream found: {e}",
            "Open Spotify and start playback.",
        )
    return DoctorCheck(
        "spotify_stream",
        "Spotify audio stream",
        "ok",
        f"{info.monitor_name} ({info.samplerate} Hz, {info.channels} ch)",
    )


def run_doctor(
    config_path: Optional[str] = None,
    *,
    process_iter: Optional[Callable[[], Iterable[Any]]] = None,
    stream_probe: Callable[[], StreamInfo] = get_spotify_stream_info,
    run_command: Callable[[List[str], float], subprocess.CompletedProcess[str]] = _run_command,
) -> DoctorReport:
    """Run first-use diagnostics without mutating recorder state."""
    config = load_user_config(config_path)
    checks = [
        _check_config(config_path),
        _check_output_dir(config),
        _check_ffmpeg(),
        _check_pactl(run_command),
        _check_pipewire_tools(),
    ]
    spotify_check = _check_spotify_process(process_iter)
    checks.append(spotify_check)
    checks.append(_check_spotify_stream(spotify_check.status == "ok", stream_probe))
    return DoctorReport(checks)
