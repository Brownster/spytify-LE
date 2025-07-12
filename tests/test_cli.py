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
    monkeypatch.setattr("spotify_splitter.main.track_events", lambda *a, **k: None)
    from spotify_splitter.util import StreamInfo
    monkeypatch.setattr(
        "spotify_splitter.main.get_spotify_stream_info",
        lambda: StreamInfo("dummy", 44100, 2),
    )
    from spotify_splitter.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    assert "Start recording until interrupted" in result.output
    assert "--player" in result.output
