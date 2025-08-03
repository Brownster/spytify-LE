import threading
import logging
from pathlib import Path
import typer
import queue
import time

from rich.live import Live
from rich.spinner import Spinner
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .audio import AudioStream, EnhancedAudioStream
from .buffer_management import AdaptiveBufferManager
from .buffer_health_monitor import BufferHealthMonitor
from .error_recovery import ErrorRecoveryManager
from .metrics_collector import MetricsCollector
from .performance_dashboard import PerformanceDashboard, DashboardConfig
from .performance_optimizer import PerformanceOptimizer
from .config_profiles import ProfileManager, ProfileType, SystemCapabilityDetector
from .segmenter import SegmentManager, OUTPUT_DIR
from .mpris import track_events
from .util import get_spotify_stream_info
from .tagging_api import tag_output
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

    For a full list of recording options, run:
    spotify-splitter record --help
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
    spotifyd_mode: bool = typer.Option(
        False,
        "--spotifyd-mode",
        help="Optimize for spotifyd usage (uses spotifyd player, headless profile).",
    ),
    profile: str = typer.Option(
        "auto",
        "--profile",
        help="Configuration profile: auto, headless, desktop, high_performance",
    ),
    enable_adaptive: bool = typer.Option(
        True,
        "--adaptive/--no-adaptive",
        help="Enable adaptive buffer management",
    ),
    enable_monitoring: bool = typer.Option(
        True,
        "--monitoring/--no-monitoring", 
        help="Enable buffer health monitoring",
    ),
    enable_metrics: bool = typer.Option(
        True,
        "--metrics/--no-metrics",
        help="Enable performance metrics collection",
    ),
    debug_mode: bool = typer.Option(
        False,
        "--debug-mode",
        help="Enable debug mode with detailed diagnostics",
    ),
    max_buffer_size: int = typer.Option(
        1000,
        "--max-buffer-size",
        help="Maximum buffer size for adaptive management",
    ),
    min_buffer_size: int = typer.Option(
        50,
        "--min-buffer-size",
        help="Minimum buffer size for adaptive management",
    ),
    playlist: str = typer.Option(
        None,
        "--playlist",
        help="Write an M3U playlist with all recorded tracks",
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
    playlist_path = Path(playlist) if playlist else None

    # Detect system capabilities and select configuration profile
    try:
        # Parse profile type
        if spotifyd_mode:
            profile_type = ProfileType.HEADLESS
            logging.info("Spotifyd mode enabled - using headless profile")
        else:
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
        
        # Override feature flags with CLI arguments
        effective_adaptive = enable_adaptive and config_profile.enable_adaptive_management
        effective_monitoring = enable_monitoring and config_profile.enable_health_monitoring
        effective_metrics = enable_metrics and config_profile.enable_metrics_collection
        effective_debug = debug_mode or config_profile.enable_debug_mode
        
        logging.info(f"Effective settings: queue_size={effective_queue_size}, blocksize={effective_blocksize}, "
                    f"latency={effective_latency}, adaptive={effective_adaptive}, monitoring={effective_monitoring}")
        
    except Exception as e:
        logging.error(f"Error configuring profile: {e}")
        # Fall back to safe defaults
        effective_queue_size = queue_size or 200
        effective_blocksize = blocksize or 2048
        effective_latency = latency or 0.1
        effective_adaptive = enable_adaptive
        effective_monitoring = enable_monitoring
        effective_metrics = enable_metrics
        effective_debug = debug_mode

    # Initialize adaptive buffer management components
    buffer_manager = None
    health_monitor = None
    error_recovery = None
    metrics_collector = None
    performance_dashboard = None
    performance_optimizer = None
    
    try:
        if effective_adaptive:
            buffer_manager = AdaptiveBufferManager(
                initial_queue_size=effective_queue_size,
                min_size=min_buffer_size,
                max_size=max_buffer_size,
                metrics_collector=None  # Will be set after metrics_collector is created
            )
            logging.info("Adaptive buffer manager initialized")
        
        if effective_monitoring and buffer_manager:
            health_monitor = BufferHealthMonitor(buffer_manager=buffer_manager)
            logging.info("Buffer health monitor initialized")
        
        error_recovery = ErrorRecoveryManager(
            max_retries=config_profile.max_reconnection_attempts if 'config_profile' in locals() else 5
        )
        logging.info("Error recovery manager initialized")
        
        if effective_metrics:
            metrics_collector = MetricsCollector(
                collection_interval=config_profile.collection_interval if 'config_profile' in locals() else 1.0,
                enable_debug_mode=effective_debug
            )
            # Link buffer manager to metrics collector
            if buffer_manager:
                buffer_manager.metrics_collector = metrics_collector
            if error_recovery:
                error_recovery.metrics_collector = metrics_collector
            logging.info("Metrics collector initialized")
            
            # Initialize performance dashboard if debug mode is enabled
            if effective_debug:
                dashboard_config = DashboardConfig(
                    update_interval=2.0,
                    enable_alerts=True,
                    enable_recommendations=True
                )
                performance_dashboard = PerformanceDashboard(
                    metrics_collector=metrics_collector,
                    config=dashboard_config
                )
                logging.info("Performance dashboard initialized")
            
            # Initialize performance optimizer
            performance_optimizer = PerformanceOptimizer(
                metrics_collector=metrics_collector,
                auto_apply_safe_optimizations=False,  # Keep manual for safety
                optimization_interval=300.0  # 5 minutes
            )
            logging.info("Performance optimizer initialized")
        
    except Exception as e:
        logging.error(f"Error initializing adaptive components: {e}")
        # Continue with basic functionality
        pass

    # Create audio queue with adaptive sizing
    audio_queue: queue.Queue = queue.Queue(maxsize=effective_queue_size)
    event_queue: queue.Queue = queue.Queue()

    # Enhanced UI state with adaptive buffer information
    ui_state = {
        "current_track": None,
        "recording_status": "Initializing adaptive buffer management...",
        "tracks_recorded": 0,
        "buffer_warnings": 0,
        "buffer_health": "Unknown",
        "buffer_utilization": 0.0,
        "adaptive_adjustments": 0,
        "emergency_expansions": 0
    }
    
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
        
        # Buffer information (if adaptive management is enabled)
        if effective_adaptive and buffer_manager:
            health_color = {
                "HEALTHY": "green",
                "WARNING": "yellow", 
                "CRITICAL": "red"
            }.get(ui_state["buffer_health"], "white")
            
            table.add_row("🔧 Buffer Health:", Text(ui_state["buffer_health"], style=health_color))
            table.add_row("📊 Buffer Usage:", f"{ui_state['buffer_utilization']:.1f}%")
            table.add_row("⚙️  Queue Size:", str(buffer_manager.current_queue_size))
            
            if ui_state["adaptive_adjustments"] > 0:
                table.add_row("🔄 Adjustments:", str(ui_state["adaptive_adjustments"]))
            
            if ui_state["emergency_expansions"] > 0:
                table.add_row("🚨 Emergency Exp:", str(ui_state["emergency_expansions"]))
        
        if ui_state["buffer_warnings"] > 0:
            table.add_row("⚠️  Buffer Warnings:", str(ui_state["buffer_warnings"]))
        
        return Panel(table, title="Spotify Splitter (Adaptive)", border_style="blue")
    
    live = Live(create_enhanced_ui(), refresh_per_second=2)
    
    def enhanced_ui_callback(action, data):
        if action == "processing":
            ui_state["recording_status"] = f"Processing: {data.title}"
        elif action == "saved":
            ui_state["tracks_recorded"] += 1
            ui_state["recording_status"] = f"Saved: {data.title}"
        elif action == "buffer_warning":
            ui_state["buffer_warnings"] += 1
        elif action == "buffer_health" and isinstance(data, dict):
            ui_state["buffer_health"] = data.get("status", "Unknown")
            ui_state["buffer_utilization"] = data.get("utilization", 0.0)
        elif action == "buffer_adjustment":
            ui_state["adaptive_adjustments"] += 1
        elif action == "emergency_expansion":
            ui_state["emergency_expansions"] += 1
        elif action == "buffer_overflow" and isinstance(data, dict):
            ui_state["buffer_warnings"] += 1
            logging.warning(f"Buffer overflow: queue_size={data.get('queue_size')}, max_size={data.get('max_size')}")
        
        # Enhanced error recovery UI callbacks
        elif action == "stream_error" and isinstance(data, dict):
            error_type = data.get("error_type", "Unknown")
            recovery_action = data.get("recovery_action", "unknown")
            ui_state["recording_status"] = f"Stream error: {error_type} - attempting {recovery_action}"
            logging.warning(f"Stream error: {error_type} -> {recovery_action}")
            
        elif action == "stream_recovery_result" and isinstance(data, dict):
            success = data.get("success", False)
            recovery_action = data.get("recovery_action", "unknown")
            error_type = data.get("error_type", "Unknown")
            
            if success:
                ui_state["recording_status"] = f"Recovery successful: {recovery_action}"
                logging.info(f"Stream recovery successful: {error_type} -> {recovery_action}")
            else:
                ui_state["recording_status"] = f"Recovery failed: {recovery_action}"
                logging.error(f"Stream recovery failed: {error_type} -> {recovery_action}")
                
        elif action == "processing_error" and isinstance(data, dict):
            track_title = data.get("track", {}).get("title", "Unknown")
            attempt = data.get("attempt", 1)
            recovery_action = data.get("recovery_action", "unknown")
            ui_state["recording_status"] = f"Processing error: {track_title} (attempt {attempt}) - {recovery_action}"
            
        elif action == "export_error" and isinstance(data, dict):
            track_title = data.get("track", {}).get("title", "Unknown")
            attempt = data.get("attempt", 1)
            ui_state["recording_status"] = f"Export error: {track_title} (attempt {attempt})"
            
        elif action == "recovery_success" and isinstance(data, dict):
            track_title = data.get("track", {}).get("title", "Unknown")
            attempts = data.get("attempts", 1)
            ui_state["recording_status"] = f"Recovery successful: {track_title} (after {attempts} attempts)"
            logging.info(f"Processing recovery successful for {track_title} after {attempts} attempts")
            
        elif action == "degraded_export" and isinstance(data, dict):
            track_title = data.get("track", {}).get("title", "Unknown")
            reason = data.get("reason", "Unknown error")
            export_type = data.get("type", "processing")
            ui_state["recording_status"] = f"Degraded {export_type}: {track_title}"
            logging.warning(f"Degraded export for {track_title}: {reason}")
            
        elif action == "processing_failure" and isinstance(data, dict):
            track = data.get("track")
            if hasattr(track, 'title'):
                track_title = track.title
            elif isinstance(track, dict):
                track_title = track.get("title", "Unknown")
            else:
                track_title = "Unknown"
            ui_state["recording_status"] = f"Processing failed: {track_title}"
            logging.error(f"Complete processing failure for {track_title}")
            
        elif action == "degraded_mode" and isinstance(data, dict):
            reason = data.get("reason", "Unknown")
            settings = data.get("settings", "Unknown settings")
            ui_state["recording_status"] = f"Degraded mode: {reason}"
            logging.warning(f"Entered degraded mode - {reason}: {settings}")
            
        elif action == "feature_degraded" and isinstance(data, dict):
            feature = data.get("feature", "unknown")
            reason = data.get("reason", "unknown")
            logging.warning(f"Feature degraded: {feature} due to {reason}")
            
        elif action == "critical_error" and isinstance(data, dict):
            error_type = data.get("error_type", "Unknown")
            ui_state["recording_status"] = f"Critical error: {error_type}"
            logging.critical(f"Critical error escalated: {error_type}")
            
            # Show recommendations if available
            recommendations = data.get("recommendations", [])
            if recommendations:
                logging.critical("Error recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    logging.critical(f"  - {rec}")
        
        live.update(create_enhanced_ui())

    manager = SegmentManager(
        samplerate=info.samplerate,
        output_dir=Path(out_dir) if out_dir else OUTPUT_DIR,
        fmt=fmt,
        audio_queue=audio_queue,
        event_queue=event_queue,
        playlist_path=playlist_path,
        ui_callback=enhanced_ui_callback,
        error_recovery=error_recovery,
        enable_error_recovery=True,
        max_processing_retries=3,
        enable_graceful_degradation=True,
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
    if spotifyd_mode:
        ui_state["recording_status"] = "Waiting for spotifyd connection..."
    else:
        ui_state["recording_status"] = "Waiting for first track..."
    
    # Reset UI state for clean startup
    ui_state["current_track"] = None
    ui_state["tracks_recorded"] = 0
    ui_state["buffer_warnings"] = 0
    ui_state["adaptive_adjustments"] = 0
    ui_state["emergency_expansions"] = 0
    
    processing_thread = threading.Thread(target=manager.run, daemon=True)

    # Start metrics collection if enabled
    if metrics_collector:
        try:
            metrics_collector.start_collection()
            logging.info("Metrics collection started")
            
            # Start performance dashboard if enabled
            if performance_dashboard:
                performance_dashboard.start_monitoring()
                logging.info("Performance dashboard started")
            
            # Start performance optimizer if enabled
            if performance_optimizer:
                performance_optimizer.start_optimization()
                logging.info("Performance optimizer started")
                
        except Exception as e:
            logging.error(f"Error starting metrics collection: {e}")

    try:
        with live:
            # Choose audio stream implementation based on adaptive management setting
            if effective_adaptive and buffer_manager:
                # Use enhanced audio stream with adaptive capabilities
                audio_stream = EnhancedAudioStream(
                    monitor_name=info.monitor_name,
                    buffer_manager=buffer_manager,
                    error_recovery=error_recovery,
                    health_monitor=health_monitor,
                    metrics_collector=metrics_collector,
                    samplerate=info.samplerate,
                    channels=info.channels,
                    q=audio_queue,
                    queue_size=effective_queue_size,
                    blocksize=effective_blocksize,
                    latency=effective_latency,
                    ui_callback=enhanced_ui_callback,
                    enable_adaptive_management=effective_adaptive,
                    enable_health_monitoring=effective_monitoring,
                    enable_metrics_collection=effective_metrics
                )
                logging.info("Using enhanced audio stream with adaptive management")
            else:
                # Use basic audio stream
                audio_stream = AudioStream(
                    info.monitor_name,
                    samplerate=info.samplerate,
                    channels=info.channels,
                    q=audio_queue,
                    queue_size=effective_queue_size,
                    blocksize=effective_blocksize,
                    latency=effective_latency,
                    ui_callback=enhanced_ui_callback,
                )
                logging.info("Using basic audio stream")
            
            # Define callbacks before audio stream context
            def on_change(track):
                logging.info(f"TRACK CHANGE CALLBACK: {track.artist} - {track.title}")
                ui_state["current_track"] = track
                if ui_state["tracks_recorded"] == 0:
                    ui_state["recording_status"] = "First track - will be discarded"
                else:
                    ui_state["recording_status"] = f"Recording: {track.artist} - {track.title}"
                live.update(create_enhanced_ui())
                event_queue.put(("track_change", track))

            def on_status(status: str):
                if status == "Playing":
                    ui_state["recording_status"] = "Recording audio..."
                elif status == "Paused":
                    ui_state["recording_status"] = "Playback paused"
                live.update(create_enhanced_ui())

            # Adjust player name for spotifyd mode
            effective_player = "spotifyd" if spotifyd_mode else player
            
            with audio_stream:
                processing_thread.start()
                
                # Start buffer health monitoring in a separate thread if enabled
                def monitor_buffer_health():
                    if effective_adaptive and buffer_manager:
                        import time
                        while processing_thread.is_alive():
                            try:
                                metrics = buffer_manager.monitor_utilization(audio_queue)
                                health = buffer_manager.get_buffer_health(metrics)
                                
                                enhanced_ui_callback("buffer_health", {
                                    "status": health.status.value.upper(),
                                    "utilization": health.utilization * 100
                                })
                                
                                time.sleep(1.0)  # Update every second
                            except Exception as e:
                                logging.debug(f"Error in buffer health monitoring: {e}")
                                time.sleep(2.0)
                
                if effective_adaptive and buffer_manager:
                    health_thread = threading.Thread(target=monitor_buffer_health, daemon=True)
                    health_thread.start()
                
                # Start MPRIS tracking in a separate thread to avoid blocking
                def mpris_wrapper():
                    try:
                        logging.info(f"Starting MPRIS tracking for player: {effective_player}")
                        track_events(
                            on_change,
                            on_status,
                            dump_metadata=dump_metadata,
                            player_name=effective_player,
                        )
                    except Exception as e:
                        logging.error(f"MPRIS tracking failed: {e}")
                        
                mpris_thread = threading.Thread(target=mpris_wrapper, daemon=True)
                mpris_thread.start()
                logging.info("MPRIS thread started")
                
                # Keep the main thread alive while MPRIS and audio recording run
                try:
                    while processing_thread.is_alive():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    raise
                
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
        # Stop performance monitoring components
        if performance_optimizer:
            try:
                performance_optimizer.stop_optimization()
                logging.info("Performance optimizer stopped")
            except Exception as e:
                logging.error(f"Error stopping performance optimizer: {e}")
        
        if performance_dashboard:
            try:
                performance_dashboard.stop_monitoring()
                logging.info("Performance dashboard stopped")
            except Exception as e:
                logging.error(f"Error stopping performance dashboard: {e}")
        
        # Stop metrics collection
        if metrics_collector:
            try:
                metrics_collector.stop_collection()
                logging.info("Metrics collection stopped")
                
                # Generate final diagnostic report if debug mode is enabled
                if effective_debug:
                    try:
                        report = metrics_collector.generate_diagnostic_report()
                        logging.info("Session performance summary:")
                        logging.info(f"  - Total metrics collected: {report.summary.get('total_metrics', 0)}")
                        logging.info(f"  - Collection uptime: {report.summary.get('collection_uptime_seconds', 0):.1f}s")
                        if report.recommendations:
                            logging.info("  - Recommendations:")
                            for rec in report.recommendations[:3]:  # Show top 3 recommendations
                                logging.info(f"    * {rec}")
                        
                        # Show optimization suggestions if available
                        if performance_optimizer:
                            suggestions = performance_optimizer.get_optimization_suggestions(limit=3)
                            if suggestions:
                                logging.info("  - Performance optimization suggestions:")
                                for suggestion in suggestions:
                                    logging.info(f"    * {suggestion.title}: {suggestion.description}")
                                    
                    except Exception as e:
                        logging.debug(f"Error generating final report: {e}")
                        
            except Exception as e:
                logging.error(f"Error stopping metrics collection: {e}")
        
        if 'manager' in locals():
            try:
                manager.close_playlist()
            except Exception as e:
                logging.debug(f"Error closing playlist: {e}")

        # Invoke tagging API on shutdown
        try:
            # Make playlist path absolute if it exists
            absolute_playlist_path = playlist_path.resolve() if playlist_path else None
            tag_output(Path(out_dir) if out_dir else OUTPUT_DIR, absolute_playlist_path)
        except Exception as e:
            logging.debug(f"Error calling tagger API: {e}")

        logging.info("Done.")


@app.command()
def profiles():
    """List available configuration profiles and system capabilities."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Display system capabilities
    console.print("\n[bold cyan]System Capabilities Detection[/bold cyan]")
    try:
        capabilities = SystemCapabilityDetector.detect_capabilities()
        
        cap_table = Table(show_header=False, box=None, padding=(0, 1))
        cap_table.add_column("Property", style="cyan")
        cap_table.add_column("Value", style="white")
        
        cap_table.add_row("CPU Cores:", str(capabilities.cpu_cores))
        cap_table.add_row("Memory:", f"{capabilities.memory_gb:.1f} GB")
        cap_table.add_row("System Load:", f"{capabilities.system_load:.1%}")
        cap_table.add_row("Headless Mode:", "Yes" if capabilities.is_headless else "No")
        cap_table.add_row("Audio Backend:", capabilities.audio_backend)
        cap_table.add_row("GUI Available:", "Yes" if capabilities.has_gui else "No")
        
        console.print(cap_table)
        
        # Show recommended profile
        recommended_profile = ProfileManager.select_optimal_profile(capabilities)
        console.print(f"\n[bold green]Recommended Profile:[/bold green] {recommended_profile.name}")
        console.print(f"[dim]{recommended_profile.description}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error detecting system capabilities: {e}[/red]")
    
    # Display available profiles
    console.print("\n[bold cyan]Available Configuration Profiles[/bold cyan]")
    
    profiles_table = Table()
    profiles_table.add_column("Profile", style="cyan", no_wrap=True)
    profiles_table.add_column("Description", style="white")
    profiles_table.add_column("Buffer Strategy", style="yellow")
    profiles_table.add_column("Queue Size", style="green")
    profiles_table.add_column("Latency", style="magenta")
    
    for profile_type in ProfileType:
        if profile_type == ProfileType.AUTO:
            continue
        
        try:
            profile = ProfileManager.get_profile(profile_type)
            profiles_table.add_row(
                profile.name,
                profile.description,
                profile.buffer_strategy.value,
                str(profile.queue_size),
                f"{profile.latency*1000:.0f}ms"
            )
        except Exception as e:
            profiles_table.add_row(
                profile_type.value,
                f"Error loading profile: {e}",
                "-", "-", "-"
            )
    
    # Add auto profile
    profiles_table.add_row(
        "auto",
        "Automatically select optimal profile based on system capabilities",
        "varies",
        "varies", 
        "varies"
    )
    
    console.print(profiles_table)
    
    console.print("\n[bold cyan]Usage Examples[/bold cyan]")
    console.print("  spotify-splitter record --profile headless")
    console.print("  spotify-splitter record --profile desktop --adaptive")
    console.print("  spotify-splitter record --profile high_performance --debug-mode")
    console.print("  spotify-splitter record --spotifyd-mode  # Uses headless profile")


if __name__ == "__main__":
    app()
