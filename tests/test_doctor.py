"""Tests for first-run doctor diagnostics."""

import subprocess

from spotify_splitter.doctor import run_doctor
from spotify_splitter.util import StreamInfo
from spotify_splitter.user_config import save_user_config


class FakeProcess:
    def __init__(self, name="spotify", cmdline=None):
        self.info = {"name": name, "cmdline": cmdline or [name]}


def ok_command(command, timeout):
    return subprocess.CompletedProcess(
        command,
        0,
        stdout="Server Name: PulseAudio (on PipeWire 1.0)\n",
        stderr="",
    )


def test_doctor_reports_ready_when_dependencies_and_stream_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    output_dir = tmp_path / "Music"
    output_dir.mkdir()
    save_user_config({"output": str(output_dir)})
    monkeypatch.setattr("spotify_splitter.doctor.shutil.which", lambda name: f"/usr/bin/{name}")

    report = run_doctor(
        process_iter=lambda: [FakeProcess()],
        stream_probe=lambda: StreamInfo("spotify.monitor", 48000, 2),
        run_command=ok_command,
    )

    assert report.ok is True
    assert report.summary == "Ready to record"
    statuses = {check.id: check.status for check in report.checks}
    assert statuses["ffmpeg"] == "ok"
    assert statuses["spotify_stream"] == "ok"


def test_doctor_spotify_running_without_stream_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("spotify_splitter.doctor.shutil.which", lambda name: f"/usr/bin/{name}")

    def no_stream():
        raise RuntimeError("Spotify sink not found")

    report = run_doctor(
        process_iter=lambda: [FakeProcess()],
        stream_probe=no_stream,
        run_command=ok_command,
    )

    checks = {check.id: check for check in report.checks}
    assert report.ok is False
    assert checks["spotify_process"].status == "ok"
    assert checks["spotify_stream"].status == "error"
    assert checks["spotify_stream"].message == "Spotify detected but not playing"
    assert "Press play" in checks["spotify_stream"].action


def test_doctor_spotify_missing_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("spotify_splitter.doctor.shutil.which", lambda name: f"/usr/bin/{name}")

    report = run_doctor(
        process_iter=lambda: [],
        stream_probe=lambda: StreamInfo("spotify.monitor", 44100, 2),
        run_command=ok_command,
    )

    spotify = {check.id: check for check in report.checks}["spotify_process"]
    assert report.ok is False
    assert spotify.status == "error"
    assert spotify.message == "Spotify not detected"


def test_doctor_reports_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def which(name):
        return None if name == "ffmpeg" else f"/usr/bin/{name}"

    monkeypatch.setattr("spotify_splitter.doctor.shutil.which", which)

    report = run_doctor(
        process_iter=lambda: [FakeProcess()],
        stream_probe=lambda: StreamInfo("spotify.monitor", 44100, 2),
        run_command=ok_command,
    )

    ffmpeg = {check.id: check for check in report.checks}["ffmpeg"]
    assert report.ok is False
    assert ffmpeg.status == "error"
    assert ffmpeg.message == "ffmpeg is missing"


def test_doctor_warns_for_missing_pipewire_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def which(name):
        return None if name == "pw-dump" else f"/usr/bin/{name}"

    monkeypatch.setattr("spotify_splitter.doctor.shutil.which", which)

    report = run_doctor(
        process_iter=lambda: [FakeProcess()],
        stream_probe=lambda: StreamInfo("spotify.monitor", 44100, 2),
        run_command=ok_command,
    )

    pipewire = {check.id: check for check in report.checks}["pipewire_tools"]
    assert pipewire.status == "warning"
    assert "Flatpak" in pipewire.message
