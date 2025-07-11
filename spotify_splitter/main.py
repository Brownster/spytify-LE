import threading
import logging
from pathlib import Path
import typer

from .audio import AudioStream
from .segmenter import SegmentManager, OUTPUT_DIR
from .mpris import track_events
from .util import find_spotify_monitor

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
    samplerate = 44100
    monitor = find_spotify_monitor()
    out_dir = ctx.obj["output"]
    fmt = ctx.obj["format"]
    manager = SegmentManager(samplerate, output_dir=Path(out_dir) if out_dir else OUTPUT_DIR, fmt=fmt)

    with AudioStream(monitor, samplerate=samplerate) as stream:
        def feeder():
            while True:
                frames = stream.read()
                manager.add_frames(frames)
        threading.Thread(target=feeder, daemon=True).start()

        def on_change(track):
            manager.start_track(track)
        track_events(on_change)


if __name__ == "__main__":
    app()
