import threading
import logging
from pathlib import Path
import typer
import queue

from rich.live import Live
from rich.spinner import Spinner
from rich.logging import RichHandler

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
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,
    )
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
def record(
    ctx: typer.Context,
    dump_metadata: bool = typer.Option(
        False,
        "--dump-metadata",
        help="Print raw MPRIS metadata for debugging.",
    ),
    player: str = typer.Option(
        "spotify",
        "--player",
        help="The MPRIS player name to connect to (e.g., 'spotify' or 'spotifyd').",
    ),
):
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

    audio_queue: queue.Queue = queue.Queue()
    event_queue: queue.Queue = queue.Queue()

    manager = SegmentManager(
        samplerate=info.samplerate,
        output_dir=Path(out_dir) if out_dir else OUTPUT_DIR,
        fmt=fmt,
        audio_queue=audio_queue,
        event_queue=event_queue,
    )

    processing_thread = threading.Thread(target=manager.run, daemon=True)

    spinner = Spinner("dots", text="Waiting for track change...")
    live = Live(spinner, transient=True, refresh_per_second=10)

    try:
        with live:
            with AudioStream(
                info.monitor_name,
                samplerate=info.samplerate,
                channels=info.channels,
                q=audio_queue,
            ):
                processing_thread.start()

                def on_change(track):
                    event_queue.put(("track_change", track))

                def on_status(status: str):
                    pass

                track_events(
                    on_change,
                    on_status,
                    dump_metadata=dump_metadata,
                    player_name=player,
                )
    except KeyboardInterrupt:
        logging.info("Shutdown signal received, processing remaining tracks...")
        event_queue.put(("shutdown", None))
        processing_thread.join()
    finally:
        logging.info("Done.")


if __name__ == "__main__":
    app()
