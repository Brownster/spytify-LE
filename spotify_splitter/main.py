import threading
import logging
from pathlib import Path
import typer

from rich.live import Live
from rich.spinner import Spinner

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

    spinner = Spinner("dots", text="Waiting for track change...")
    live = Live(spinner, transient=True, refresh_per_second=10)

    try:
        with live:
            # The ``AudioStream`` uses a high priority callback that places
            # captured frames into an internal queue. A dedicated thread simply
            # pulls from that queue and hands the data off to the segment
            # manager. This avoids busy waiting and reduces the chance of
            # buffer overruns.
            with AudioStream(
                info.monitor_name,
                samplerate=info.samplerate,
                channels=info.channels,
            ) as stream:

                def audio_processor():
                    """Continuously read frames from ``stream`` and buffer them."""
                    while True:
                        frames = stream.read()
                        manager.add_frames(frames)

                threading.Thread(target=audio_processor, daemon=True).start()

                def on_change(track):
                    manager.start_track(track)
                    if manager.current:
                        spinner.text = f"Recording: [bold cyan]{track.artist} – {track.title}[/]"
                    else:
                        spinner.text = "Skipping ad..."

                def on_status(status: str):
                    if status == "Paused":
                        manager.pause_recording()
                        spinner.text = "[bold yellow]Paused[/]"
                    elif status == "Playing":
                        manager.resume_recording()
                        if manager.current:
                            spinner.text = f"Recording: [bold cyan]{manager.current.artist} – {manager.current.title}[/]"
                        else:
                            spinner.text = "Waiting for track change..."

                track_events(on_change, on_status)
    except KeyboardInterrupt:
        logging.info("Recording interrupted by user.")
    finally:
        logging.info("Saving final track...")
        manager.flush()
        logging.info("Done.")


if __name__ == "__main__":
    app()
