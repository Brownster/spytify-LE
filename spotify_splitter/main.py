import threading
import logging
from pathlib import Path
import typer

from .audio import AudioStream
from .segmenter import SegmentManager, OUTPUT_DIR
from .mpris import track_events
from .util import get_spotify_stream_info
try:
    from pydbus.errors import DBusError
except Exception:  # pragma: no cover - fallback if gi is missing
    class DBusError(Exception):
        pass

app = typer.Typer(add_completion=False)


@app.callback()
def main_callback(
    ctx: typer.Context,
    output: str = typer.Option(None, help="Directory to save tracks"),
    format: str = typer.Option("mp3", help="Output format: mp3, flac, etc."),
    verbose: bool = typer.Option(False, "-v", help="Enable debug logging"),
):
    """Record Spotify playback and split into tracks."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, handlers=[logging.StreamHandler()])
    if output:
        ctx.obj = {
            "output": output,
            "format": format,
        }
    else:
        ctx.obj = {
            "output": None,
            "format": format,
        }


@app.command()
def record(ctx: typer.Context):
    """Start recording until interrupted."""
    try:
        info = get_spotify_stream_info()
    except RuntimeError as e:
        logging.error(f"Error finding audio source: {e}")
        raise typer.Exit(code=1)
    except DBusError as e:
        logging.error(f"D-Bus error: {e}. Is Spotify running?")
        raise typer.Exit(code=1)

    out_dir = ctx.obj["output"]
    fmt = ctx.obj["format"]
    manager = SegmentManager(info.samplerate, output_dir=Path(out_dir) if out_dir else OUTPUT_DIR, fmt=fmt)

    try:
        with AudioStream(info.monitor_name, samplerate=info.samplerate, channels=info.channels) as stream:
            def feeder():
                while True:
                    frames = stream.read()
                    manager.add_frames(frames)

            threading.Thread(target=feeder, daemon=True).start()

            def on_change(track):
                manager.start_track(track)

            def on_status(status: str):
                if status == "Paused":
                    manager.pause_recording()
                elif status == "Playing":
                    manager.resume_recording()

            track_events(on_change, on_status)
    except KeyboardInterrupt:
        logging.info("Recording interrupted by user.")
    finally:
        logging.info("Saving final track...")
        manager.flush()
        logging.info("Done.")


if __name__ == "__main__":
    app()
