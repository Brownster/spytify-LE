import sys
import types
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
    monkeypatch.setattr("spotify_splitter.main.track_events", lambda func: None)
    monkeypatch.setattr("spotify_splitter.main.find_spotify_monitor", lambda: "dummy")
    from spotify_splitter.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    assert "Start recording until interrupted" in result.output
