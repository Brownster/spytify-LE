import threading
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import typer
import queue
import time

from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from click.core import ParameterSource

from .audio import AudioStream
from .config_profiles import ProfileManager, ProfileType
from .engine import RecorderConfigError, RecorderEngine, RecorderEngineConfig, RecorderError
from .segmenter import SegmentManager, OUTPUT_DIR
from .mpris import track_events
from .recorder_status import (
    AtomicStatusWriter,
    AudioStatus,
    RecorderStatus,
    TimerStatus,
    TrackStatus,
)
from .track_history import TrackHistoryWriter
from .util import get_spotify_stream_info, stream_info_for_source
from .tagging_api import tag_output
from .user_config import (
    DEFAULT_CONFIG,
    load_user_config,
)
from .cli_commands import register as register_cli_commands
try:
    from pydbus.errors import DBusError
except Exception:  # pragma: no cover - fallback if gi is missing
    class DBusError(Exception):
        pass

app = typer.Typer(add_completion=False)
_LAST_ERROR_UNSET = object()
STATUS_HEARTBEAT_INTERVAL_SECONDS = 5.0

# The auxiliary `profiles` and `configure` commands live in cli_commands.
register_cli_commands(app)


@app.callback()
def main_callback(
    ctx: typer.Context,
    output: Optional[str] = typer.Option(None, help="Directory to save tracks"),
    format: Optional[str] = typer.Option(None, help="Output format: mp3, flac, etc."),
    verbose: bool = typer.Option(False, "-v", help="Enable debug logging"),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (defaults to ~/.config/spotify_splitter/config.json)",
    ),
):
    """Record Spotify desktop playback and split into tracks.

    Monitors Linux Spotify desktop client via MPRIS and records audio
    via PulseAudio/PipeWire. Automatically tags with LastFM metadata.

    For a full list of recording options, run:
    spotify-splitter record --help
    """
    level = logging.DEBUG if verbose else logging.INFO
    # Use a simple format when running under service to avoid duplicate prefixes
    if os.environ.get("RICH_FORCE_TERMINAL") == "0":
        # Running as subprocess - use simple format
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s",
            force=True,
        )
    else:
        # Running interactively - use RichHandler
        logging.basicConfig(
            level=level,
            handlers=[RichHandler(rich_tracebacks=True)],
            force=True,
        )
    config = load_user_config(config_path)
    resolved_output = Path(output).expanduser() if output else Path(config["output"]).expanduser()
    resolved_format = format or config.get("format", "mp3")

    ctx.obj = {
        "output": str(resolved_output),
        "format": resolved_format,
        "config": config,
        "config_path": config_path,
        "verbose": verbose,
    }


@app.command()
def record(
    ctx: typer.Context,
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Directory to save tracks (also accepted as a global option)",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        help="Output format: mp3, flac, etc. (also accepted as a global option)",
    ),
    dump_metadata: bool = typer.Option(
        False,
        "--dump-metadata",
        help="Print raw MPRIS metadata for debugging.",
    ),
    player: str = typer.Option(
        "spotify",
        "--player",
        help="The MPRIS player name (usually 'spotify' for Spotify desktop).",
    ),
    spotifyd_mode: bool = typer.Option(
        False,
        "--spotifyd-mode",
        help="Use spotifyd/headless defaults (player=spotifyd, profile=headless unless overridden).",
    ),
    queue_size: int = typer.Option(
        None,
        "--queue-size",
        help="Audio buffer size (number of blocks). If not specified, uses profile default.",
    ),
    blocksize: int = typer.Option(
        None,
        "--blocksize",
        help="Number of frames per audio callback. If not specified, uses profile default.",
    ),
    latency: float = typer.Option(
        None,
        "--latency",
        help="Desired latency for the audio stream in seconds. If not specified, uses profile default.",
    ),
    profile: str = typer.Option(
        "auto",
        "--profile",
        help="Configuration profile: auto, headless, desktop, high_performance",
    ),
    playlist: str = typer.Option(
        None,
        "--playlist",
        help="Write an M3U playlist with all recorded tracks",
    ),
    bundle_playlist: bool = typer.Option(
        False,
        "--bundle-playlist",
        help="Use playlist name as album and tag album artist as 'Various Artists'",
    ),
    bundle_album_art_uri: str = typer.Option(
        None,
        "--bundle-album-art-uri",
        help="Custom album artwork URL for bundle playlists (defaults to first track's artwork if not provided)",
    ),
    playlist_base_path: str = typer.Option(
        None,
        "--playlist-base-path",
        help="Base path for M3U playlist entries (e.g., NAS mount point like '/mnt/storage/music'). Allows mapping local recording paths to remote server paths.",
    ),
    max_duration: str = typer.Option(
        None,
        "--max-duration",
        "--duration",
        help="Maximum recording duration (e.g., '4h29m', '2h30m', '90m', '5400s'). Recording stops automatically when timer expires.",
    ),
    status_file: str = typer.Option(
        None,
        "--status-file",
        help="Write structured recorder status to this JSON file using atomic replace.",
    ),
    history_file: str = typer.Option(
        None,
        "--history-file",
        help="Append per-track recording outcomes (saved/skipped/failed + year/genre) "
        "to this capped JSONL file.",
    ),
    control_stdin: bool = typer.Option(
        False,
        "--control-stdin",
        help="Read newline-delimited JSON control commands from stdin.",
    ),
    monitor: str = typer.Option(
        None,
        "--monitor",
        "--source",
        help="Capture source/device name to record from, overriding auto-detection. "
        "Accepts a PortAudio input device or a PulseAudio source. "
        "List candidates with: pactl list sources short  (or check 'pactl list sink-inputs' node.name).",
    ),
):
    """Start recording until interrupted."""
    config = ctx.obj.get("config", DEFAULT_CONFIG.copy())

    if output:
        ctx.obj["output"] = str(Path(output).expanduser())
    if format:
        ctx.obj["format"] = format

    def resolve_param(name: str, current_value, config_key: Optional[str] = None):
        """Prefer CLI value when explicitly provided, otherwise fall back to config."""
        key = config_key or name
        source = ctx.get_parameter_source(name)
        if source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
            return config.get(key, current_value)
        return current_value

    dump_metadata = resolve_param("dump_metadata", dump_metadata)
    player = resolve_param("player", player)
    spotifyd_mode = resolve_param("spotifyd_mode", spotifyd_mode)
    queue_size = resolve_param("queue_size", queue_size)
    blocksize = resolve_param("blocksize", blocksize)
    latency = resolve_param("latency", latency)
    profile = resolve_param("profile", profile)
    playlist = resolve_param("playlist", playlist)
    bundle_playlist = resolve_param("bundle_playlist", bundle_playlist)
    bundle_album_art_uri = resolve_param("bundle_album_art_uri", bundle_album_art_uri)
    playlist_base_path = resolve_param("playlist_base_path", playlist_base_path)
    max_duration = resolve_param("max_duration", max_duration)
    status_file = resolve_param("status_file", status_file)
    history_file = resolve_param("history_file", history_file)
    control_stdin = resolve_param("control_stdin", control_stdin)
    monitor = resolve_param("monitor", monitor)

    if spotifyd_mode:
        player = "spotifyd"
        if profile == "auto":
            profile = "headless"

    # Validate max_duration format if provided
    if max_duration:
        from .duration_parser import parse_duration, format_remaining_time
        try:
            timer_duration = parse_duration(max_duration)
            if timer_duration < 60:
                logging.warning(
                    f"Timer duration very short ({timer_duration}s) - first track may be incomplete"
                )
            logging.info(f"Recording timer set for {format_remaining_time(timer_duration)}")
        except ValueError as e:
            logging.error(f"Invalid duration format '{max_duration}': {e}")
            typer.echo(
                "Error: Invalid duration format. Examples: '4h29m', '2h30m', '90m', '5400s'",
                err=True
            )
            raise typer.Exit(code=1)

    try:
        if monitor:
            info = stream_info_for_source(monitor)
        else:
            info = get_spotify_stream_info()
    except RuntimeError as e:
        logging.error(f"Error finding audio source: {e}")
        raise typer.Exit(code=1)
    except DBusError as e:
        logging.error(f"D-Bus error: {e}. Is Spotify running?")
        raise typer.Exit(code=1)

    out_dir = ctx.obj["output"]
    fmt = ctx.obj["format"]
    playlist_path = Path(playlist) if playlist else None
    if bundle_playlist and not playlist_path:
        logging.error("--bundle-playlist requires --playlist to be set")
        raise typer.Exit(code=1)

    # Detect system capabilities and select configuration profile
    try:
        # Parse profile type
        try:
            profile_type = ProfileType(profile.lower())
        except ValueError:
            logging.warning(f"Unknown profile '{profile}', using auto selection")
            profile_type = ProfileType.AUTO
        
        # Get configuration profile
        config_profile = ProfileManager.get_profile(profile_type)
        logging.info(f"Using configuration profile: {config_profile.name} - {config_profile.description}")
        
        # Override profile settings with CLI arguments if provided
        effective_queue_size = queue_size if queue_size is not None else config_profile.queue_size
        effective_blocksize = blocksize if blocksize is not None else config_profile.blocksize
        effective_latency = latency if latency is not None else config_profile.latency

        logging.info(f"Effective settings: queue_size={effective_queue_size}, "
                    f"blocksize={effective_blocksize}, latency={effective_latency}")

    except Exception as e:
        logging.error(f"Error configuring profile: {e}")
        # Fall back to safe defaults
        effective_queue_size = queue_size or 200
        effective_blocksize = blocksize or 2048
        effective_latency = latency or 0.1

    # Get allow_overwrite and lastfm_api_key from config
    allow_overwrite = config.get("allow_overwrite", False)
    lastfm_api_key = config.get("lastfm_api_key")

    try:
        engine_config = RecorderEngineConfig(
            stream_info=info,
            output_dir=Path(out_dir) if out_dir else OUTPUT_DIR,
            fmt=fmt,
            player=player,
            dump_metadata=dump_metadata,
            queue_size=effective_queue_size,
            blocksize=effective_blocksize,
            latency=effective_latency,
            playlist_path=playlist_path,
            bundle_playlist=bundle_playlist,
            bundle_album_art_uri=bundle_album_art_uri,
            playlist_base_path=playlist_base_path,
            max_duration=max_duration,
            timer_duration_seconds=timer_duration if max_duration else None,
            allow_overwrite=allow_overwrite,
            lastfm_api_key=lastfm_api_key,
            status_file=Path(status_file) if status_file else None,
            control_stdin=control_stdin,
        )
    except RecorderConfigError as e:
        logging.error(f"Invalid recorder configuration: {e}")
        raise typer.Exit(code=1)

    engine = RecorderEngine(engine_config)
    audio_queue = engine.audio_queue
    event_queue = engine.event_queue

    ui_state = {
        "current_track": None,
        "recording_status": "Initializing...",
        "tracks_recorded": 0,
        "buffer_warnings": 0,
        # Timer state
        "timer_enabled": False,
        "timer_duration_seconds": 0,
        "timer_elapsed_seconds": 0,
        "timer_remaining_seconds": 0,
    }

    status_writer = AtomicStatusWriter(status_file, fsync=False) if status_file else None
    history_writer = TrackHistoryWriter(history_file) if history_file else None
    recorder_status = RecorderStatus(state="starting")
    # Session-constant capture facts for the UI's now-playing card.
    recorder_status.samplerate = info.samplerate
    recorder_status.output_format = fmt
    status_lock = threading.Lock()
    last_metric_status_publish = 0.0

    def publish_status(state: Optional[str] = None, last_error=_LAST_ERROR_UNSET) -> None:
        nonlocal last_metric_status_publish
        if not status_writer:
            return
        is_metric_only = state is None and last_error is _LAST_ERROR_UNSET
        now = time.monotonic()
        if is_metric_only and now - last_metric_status_publish < 1.0:
            return

        with status_lock:
            if state:
                recorder_status.state = state
            if last_error is not _LAST_ERROR_UNSET:
                recorder_status.last_error = last_error
            if ui_state["current_track"]:
                recorder_status.current_track = TrackStatus.from_track_info(ui_state["current_track"])
            recorder_status.tracks_recorded = ui_state["tracks_recorded"]
            recorder_status.timer = TimerStatus(
                enabled=ui_state["timer_enabled"],
                elapsed_seconds=ui_state["timer_elapsed_seconds"],
                remaining_seconds=ui_state["timer_remaining_seconds"],
            )
            recorder_status.audio = AudioStatus(
                queue_depth=audio_queue.qsize(),
                # TODO(Pass 2): populate from real callback drop accounting.
                dropped_frames=0,
                buffer_warnings=ui_state["buffer_warnings"],
            )
            try:
                status_writer.write(recorder_status)
                if is_metric_only:
                    last_metric_status_publish = now
            except Exception as e:
                logging.debug(f"Error writing recorder status file: {e}")

    def apply_timer_snapshot(snapshot) -> None:
        ui_state["timer_enabled"] = snapshot.enabled
        ui_state["timer_duration_seconds"] = snapshot.duration_seconds
        ui_state["timer_elapsed_seconds"] = snapshot.elapsed_seconds
        ui_state["timer_remaining_seconds"] = snapshot.remaining_seconds

    def on_timer_tick(timer_tick) -> None:
        apply_timer_snapshot(timer_tick.snapshot)
        publish_status("recording")

    def on_timer_expired(timer_tick) -> None:
        apply_timer_snapshot(timer_tick.snapshot)
        logging.info("Recording timer expired - initiating graceful shutdown")
        ui_state["recording_status"] = "Timer expired - stopping recording..."
        publish_status("stopping")
    
    def create_enhanced_ui():
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

        # Timer information (if enabled)
        if ui_state["timer_enabled"]:
            from .duration_parser import format_remaining_time

            remaining = ui_state["timer_remaining_seconds"]
            elapsed = ui_state["timer_elapsed_seconds"]
            total = ui_state["timer_duration_seconds"]

            # Calculate progress percentage
            progress_pct = (elapsed / total * 100) if total > 0 else 0

            # Color based on remaining time
            if remaining > 600:  # > 10 minutes
                time_color = "green"
            elif remaining > 300:  # > 5 minutes
                time_color = "yellow"
            else:
                time_color = "red"

            table.add_row("⏱️  Timer:", Text(format_remaining_time(remaining), style=time_color))
            table.add_row("⏳ Progress:", f"{progress_pct:.1f}% ({format_remaining_time(elapsed)} / {format_remaining_time(total)})")

        # Update panel title to indicate timer mode
        title = "Spotify Splitter (Timed Recording)" if ui_state["timer_enabled"] else "Spotify Splitter"
        return Panel(table, title=title, border_style="blue")
    
    def enhanced_ui_callback(action, data):
        if action == "processing":
            ui_state["recording_status"] = f"Processing: {data.title}"
            publish_status("processing")
        elif action == "saved":
            ui_state["tracks_recorded"] += 1
            ui_state["recording_status"] = f"Saved: {data.title}"
            publish_status("recording")
        elif action == "buffer_warning":
            ui_state["buffer_warnings"] += 1
            publish_status()

        elif action == "processing_error" and isinstance(data, dict):
            track_title = data.get("track", {}).get("title", "Unknown")
            attempt = data.get("attempt", 1)
            recovery_action = data.get("recovery_action", "unknown")
            ui_state["recording_status"] = f"Processing error: {track_title} (attempt {attempt}) - {recovery_action}"
            publish_status("error", f"{track_title}: {recovery_action}")
            
        elif action == "export_error" and isinstance(data, dict):
            track_title = data.get("track", {}).get("title", "Unknown")
            attempt = data.get("attempt", 1)
            ui_state["recording_status"] = f"Export error: {track_title} (attempt {attempt})"
            publish_status("error", f"{track_title}: export attempt {attempt}")

        elif action == "processing_failure" and isinstance(data, dict):
            track = data.get("track")
            error_msg = data.get("error", "Processing failed")
            if hasattr(track, 'title'):
                track_title = track.title
            elif isinstance(track, dict):
                track_title = track.get("title", "Unknown")
            else:
                track_title = "Unknown"
            ui_state["recording_status"] = f"Failed: {track_title}"
            logging.error(f"Processing failure for {track_title}: {error_msg}")
            publish_status("error", f"{track_title}: {error_msg}")

    manager = SegmentManager(
        samplerate=info.samplerate,
        output_dir=engine_config.output_dir,
        fmt=engine_config.fmt,
        audio_queue=audio_queue,
        event_queue=event_queue,
        playlist_path=engine_config.playlist_path,
        bundle_playlist=engine_config.bundle_playlist,
        bundle_album_art_uri=engine_config.bundle_album_art_uri,
        playlist_base_path=engine_config.playlist_base_path,
        ui_callback=enhanced_ui_callback,
        allow_overwrite=engine_config.allow_overwrite,
        lastfm_api_key=engine_config.lastfm_api_key,
        on_track_result=history_writer.append if history_writer else None,
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
    
    # Update UI state for startup
    ui_state["recording_status"] = "Waiting for first track..."
    
    # Reset UI state for clean startup
    ui_state["current_track"] = None
    ui_state["tracks_recorded"] = 0
    ui_state["buffer_warnings"] = 0
    engine.set_status_publisher(publish_status)
    engine.configure_post_run_cleanup(tag_output=tag_output)
    publish_status("waiting")

    def graceful_shutdown() -> None:
        engine.stop(flush=True)

    def on_control_stop_requested() -> None:
        ui_state["recording_status"] = "Stop requested - finalizing recording..."

    def create_audio_stream():
        return AudioStream(
            info.monitor_name,
            samplerate=info.samplerate,
            channels=info.channels,
            q=audio_queue,
            queue_size=effective_queue_size,
            blocksize=effective_blocksize,
            latency=effective_latency,
            ui_callback=enhanced_ui_callback,
        )

    def on_change(track):
        logging.info(f"TRACK CHANGE CALLBACK: {track.artist} - {track.title}")
        ui_state["current_track"] = track
        if ui_state["tracks_recorded"] == 0:
            ui_state["recording_status"] = "First track - will be discarded"
        else:
            ui_state["recording_status"] = f"Recording: {track.artist} - {track.title}"
        publish_status("recording")
        event_queue.put(("track_change", track))

    def on_status(status: str):
        if status == "Playing":
            ui_state["recording_status"] = "Recording audio..."
            publish_status("recording")
        elif status == "Paused":
            ui_state["recording_status"] = "Playback paused"
            publish_status("paused")

    def run_track_events(on_track_change, on_playback_status):
        track_events(
            on_track_change,
            on_playback_status,
            dump_metadata=dump_metadata,
            player_name=player,
        )

    engine_start_kwargs = {
        "manager": manager,
        "processing_target": manager.run,
        "stream_factory": create_audio_stream,
        "track_event_runner": run_track_events,
        "on_track_change": on_change,
        "on_playback_status": on_status,
        "control_input_stream": sys.stdin if control_stdin else None,
        "on_control_stop_requested": on_control_stop_requested,
        "heartbeat_interval": STATUS_HEARTBEAT_INTERVAL_SECONDS,
        "on_heartbeat": publish_status,
        "on_timer_tick": on_timer_tick,
        "on_timer_expired": on_timer_expired,
    }

    try:
        headless_mode = os.environ.get("RICH_FORCE_TERMINAL") == "0"
        if headless_mode:
            try:
                engine.run(**engine_start_kwargs)
            except RecorderError as e:
                logging.error("Recorder failed to start: %s", e)
                raise typer.Exit(code=1)
        else:
            live = Live(create_enhanced_ui(), refresh_per_second=2)
            with live:
                try:
                    engine.start(**engine_start_kwargs)
                except RecorderError as e:
                    logging.error("Recorder failed to start: %s", e)
                    raise typer.Exit(code=1)

                # Keep the main thread alive while MPRIS and audio recording run
                # Initialize timer display if max_duration is specified
                if engine.is_timer_enabled():
                    apply_timer_snapshot(engine.timer_snapshot())
                    publish_status("recording")

                try:
                    while not engine.is_stopped():
                        live.update(create_enhanced_ui())
                        time.sleep(0.1)

                except KeyboardInterrupt:
                    raise
                
    except KeyboardInterrupt:
        logging.info("Shutdown signal received")
        graceful_shutdown()
        
    finally:
        if 'manager' in locals():
            try:
                graceful_shutdown()
            except Exception as e:
                logging.debug(f"Error during shutdown cleanup: {e}")

        if 'engine' in locals():
            engine.finalize_post_run()

        logging.info("Done.")


if __name__ == "__main__":
    app()
