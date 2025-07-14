import threading
import logging
from pathlib import Path
import typer
import queue

from rich.live import Live
from rich.spinner import Spinner
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

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
    """Record Spotify playback and split into tracks.
    
    Supports both regular Spotify client and headless spotifyd usage.
    Use --spotifyd-mode for optimized headless operation.
    """
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
    queue_size: int = typer.Option(
        200,
        "--queue-size",
        help="Audio buffer size (number of blocks) before frames are dropped.",
    ),
    blocksize: int = typer.Option(
        None,
        "--blocksize",
        help="Number of frames per audio callback.",
    ),
    latency: float = typer.Option(
        None,
        "--latency",
        help="Desired latency for the audio stream in seconds.",
    ),
    spotifyd_mode: bool = typer.Option(
        False,
        "--spotifyd-mode",
        help="Optimize for spotifyd usage (uses spotifyd player, larger buffers, higher latency tolerance).",
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

    # Create shared state for UI updates
    ui_state = {
        "current_track": None,
        "recording_status": "Waiting for first track...",
        "tracks_recorded": 0,
        "buffer_warnings": 0
    }
    
    def create_ui():
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="white")
        
        # Current track info
        if ui_state["current_track"]:
            track = ui_state["current_track"]
            table.add_row("🎵 Current Track:", f"{track.artist} - {track.title}")
            table.add_row("📀 Album:", track.album)
        
        # Recording status
        status_color = "green" if "Recording" in ui_state["recording_status"] else "yellow"
        table.add_row("📊 Status:", Text(ui_state["recording_status"], style=status_color))
        
        # Stats
        table.add_row("💾 Tracks Recorded:", str(ui_state["tracks_recorded"]))
        
        if ui_state["buffer_warnings"] > 0:
            table.add_row("⚠️  Buffer Warnings:", str(ui_state["buffer_warnings"]))
        
        return Panel(table, title="Spotify Splitter", border_style="blue")
    
    live = Live(create_ui(), refresh_per_second=2)
    
    def ui_update_callback(action, track_info):
        if action == "processing":
            ui_state["recording_status"] = f"Processing: {track_info.title}"
        elif action == "saved":
            ui_state["tracks_recorded"] += 1
            ui_state["recording_status"] = f"Saved: {track_info.title}"
        elif action == "buffer_warning":
            ui_state["buffer_warnings"] += 1
        live.update(create_ui())

    manager = SegmentManager(
        samplerate=info.samplerate,
        output_dir=Path(out_dir) if out_dir else OUTPUT_DIR,
        fmt=fmt,
        audio_queue=audio_queue,
        event_queue=event_queue,
        ui_callback=ui_update_callback,
    )
    
    # Flush any cached data from previous runs
    logging.info("Performing startup cleanup - flushing cache for clean start...")
    manager.flush_cache()
    logging.info("Startup cleanup complete - ready to record")
    
    # Clear any leftover data in fresh queues (shouldn't be any, but safety check)
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except queue.Empty:
            break
    
    # Apply spotifyd mode optimizations
    if spotifyd_mode:
        logging.info("Spotifyd mode enabled - applying optimizations...")
        # Increase buffer sizes for headless operation
        queue_size = max(queue_size, 300)
        blocksize = blocksize or 4096  # Larger blocks for stability
        latency = latency or 'high'     # Higher latency tolerance
        ui_state["recording_status"] = "Waiting for spotifyd connection..."
    else:
        ui_state["recording_status"] = "Waiting for first track..."
    
    # Reset UI state for clean startup
    ui_state["current_track"] = None
    ui_state["tracks_recorded"] = 0
    ui_state["buffer_warnings"] = 0
    
    processing_thread = threading.Thread(target=manager.run, daemon=True)

    try:
        with live:
            with AudioStream(
                info.monitor_name,
                samplerate=info.samplerate,
                channels=info.channels,
                q=audio_queue,
                queue_size=queue_size,
                blocksize=blocksize,
                latency=latency,
                ui_callback=ui_update_callback,
            ):
                processing_thread.start()

                def on_change(track):
                    ui_state["current_track"] = track
                    if ui_state["tracks_recorded"] == 0:
                        ui_state["recording_status"] = "First track - will be discarded"
                    else:
                        ui_state["recording_status"] = f"Recording: {track.artist} - {track.title}"
                    live.update(create_ui())
                    event_queue.put(("track_change", track))

                def on_status(status: str):
                    if status == "Playing":
                        ui_state["recording_status"] = "Recording audio..."
                    elif status == "Paused":
                        ui_state["recording_status"] = "Playback paused"
                    live.update(create_ui())

                # Adjust player name for spotifyd mode
                effective_player = "spotifyd" if spotifyd_mode else player
                
                track_events(
                    on_change,
                    on_status,
                    dump_metadata=dump_metadata,
                    player_name=effective_player,
                )
    except KeyboardInterrupt:
        logging.info("Shutdown signal received, processing remaining tracks...")
        
        # Perform cleanup - process any remaining tracks
        manager.shutdown_cleanup()
        
        # Signal shutdown to processing thread
        event_queue.put(("shutdown", None))
        processing_thread.join()
        
        # Final cache flush
        manager.flush_cache()
    finally:
        logging.info("Done.")


if __name__ == "__main__":
    app()
