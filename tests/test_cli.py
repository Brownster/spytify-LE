from typer.testing import CliRunner
import sys

class DummySD:
    pass

class DummyGI:
    class repository:
        class GLib:
            pass

class DummyDBus:
    class SessionBus:
        def get(self, *args, **kwargs):
            class Dummy:
                def __setattr__(self, name, value):
                    pass
            return Dummy()

sys.modules.setdefault('sounddevice', DummySD())
sys.modules.setdefault('gi', DummyGI())
sys.modules.setdefault('gi.repository', DummyGI.repository)
sys.modules.setdefault('pydbus', DummyDBus())
from spotify_splitter.main import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "record" in result.output
