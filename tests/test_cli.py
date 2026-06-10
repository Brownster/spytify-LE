import sys
import types
import re
from typer.testing import CliRunner
import pytest


@pytest.fixture(autouse=True)
def dummy_modules(monkeypatch):
    # Provide lightweight stand-ins for optional C libraries
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    gi = types.ModuleType("gi")
    gi.repository = types.SimpleNamespace(GLib=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", gi.repository)
    dbus = types.ModuleType("pydbus")
    dbus.SessionBus = lambda: None
    monkeypatch.setitem(sys.modules, "pydbus", dbus)


def test_cli_help(monkeypatch):
    monkeypatch.setattr("spotify_splitter.main.track_events", lambda *a, **k: None)
    from spotify_splitter.util import StreamInfo
    monkeypatch.setattr(
        "spotify_splitter.main.get_spotify_stream_info",
        lambda: StreamInfo("dummy", 44100, 2),
    )
    from spotify_splitter.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["record", "--help"], env={"COLUMNS": "140"})
    assert result.exit_code == 0

    # Strip ANSI escape sequences which may be present when color output is
    # enabled in CI environments.
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    output = ansi_escape.sub("", result.output)

    assert "Start recording until interrupted" in output
    assert "--player" in output
    assert "--queue-size" in output
    assert "--blocksize" in output
    assert "--latency" in output
    assert "--playlist" in output
    assert "--duration" in output


def test_doctor_command_json(monkeypatch):
    from spotify_splitter.main import app

    class Report:
        ok = True

        def to_dict(self):
            return {"ok": True, "summary": "Ready to record", "checks": []}

    monkeypatch.setattr("spotify_splitter.cli_commands.run_doctor", lambda config_path=None: Report())

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    assert '"summary": "Ready to record"' in result.output


def test_web_command_starts_local_service(monkeypatch):
    from spotify_splitter.main import app

    calls = []
    monkeypatch.setattr(
        "spoti2_service.service_app.run_service",
        lambda **kwargs: calls.append(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["web"])

    assert result.exit_code == 0
    assert calls == [{
        "host": "127.0.0.1",
        "port": 8730,
        "config": None,
        "verbose": False,
        "open_browser": True,
    }]


def test_web_command_can_disable_browser(monkeypatch):
    from spotify_splitter.main import app

    calls = []
    monkeypatch.setattr(
        "spoti2_service.service_app.run_service",
        lambda **kwargs: calls.append(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["web", "--no-open", "--port", "9999", "--config", "/tmp/cfg.json"])

    assert result.exit_code == 0
    assert calls[0]["port"] == 9999
    assert calls[0]["open_browser"] is False
    assert calls[0]["config"] == "/tmp/cfg.json"
