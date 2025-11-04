import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs

from spotify_splitter.user_config import (
    DEFAULT_CONFIG,
    get_config_path,
    load_user_config,
    save_user_config,
)


LOG_DIR = Path.home() / ".cache" / "spotify_splitter" / "service"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "service.log"


def configure_logging(verbose: bool = False) -> None:
    """Configure logging to file and stdout."""
    handlers = [
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


class RecorderSupervisor:
    """Keep spotify-splitter running and restart on failure or config changes."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        auto_restart_delay: float = 15.0,
        disable_metrics: bool = True,
    ) -> None:
        self.config_path = config_path
        self.auto_restart_delay = auto_restart_delay
        self.disable_metrics = disable_metrics

        self._process: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._pause_event = threading.Event()
        self._manual_stop = False  # Track if user manually stopped recording
        self._status_lock = threading.Lock()
        self._status: Dict[str, str] = {
            "state": "stopped",
            "details": "Click Start to begin recording",
            "last_exit": "",
            "current_track": "",
        }
        self._verbose_logging = False  # Toggle for minimal vs verbose logging

    # Public API -------------------------------------------------------------
    def start(self) -> None:
        """Start or resume recording."""
        self._manual_stop = False
        if self._thread and self._thread.is_alive():
            if self._pause_event.is_set():
                # Resume from pause
                logging.info("Resuming recording")
                self._pause_event.clear()
                self._set_status("running", "Recording resumed")
            else:
                # Restart if stopped
                logging.info("Restarting recording")
                self._restart_event.set()
            return

        logging.info("Starting RecorderSupervisor thread")
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop recording (user-initiated)."""
        logging.info("Stopping RecorderSupervisor (manual stop)")
        self._manual_stop = True
        self._stop_event.set()
        self._restart_event.set()
        self._pause_event.clear()
        self._terminate_process()
        self._set_status("stopped", "⏹️ Recording stopped")

    def pause(self) -> None:
        """Pause recording."""
        if self._process and not self._pause_event.is_set():
            logging.info("Pausing recording")
            self._pause_event.set()
            try:
                self._process.send_signal(signal.SIGSTOP)
                self._set_status("paused", "⏸️ Recording paused")
            except Exception as e:
                logging.warning("Failed to pause process: %s", e)
                self._set_status("error", f"❌ Failed to pause: {e}")
        else:
            logging.debug("Cannot pause - no active recording")

    def resume(self) -> None:
        """Resume from pause."""
        if self._process and self._pause_event.is_set():
            logging.info("Resuming recording")
            self._pause_event.clear()
            try:
                self._process.send_signal(signal.SIGCONT)
                self._set_status("running", "▶️ Recording resumed")
            except Exception as e:
                logging.warning("Failed to resume process: %s", e)
                self._set_status("error", f"❌ Failed to resume: {e}")
        else:
            logging.debug("Cannot resume - not paused")

    def request_restart(self, reason: str = "Configuration updated") -> None:
        logging.info("Restart requested: %s", reason)
        self._restart_event.set()
        self._terminate_process()
        self._set_status("restarting", reason)

    def status(self) -> Dict[str, str]:
        with self._status_lock:
            return dict(self._status)

    def set_verbose_logging(self, enabled: bool) -> None:
        """Toggle verbose logging mode."""
        self._verbose_logging = enabled
        logging.info("Verbose logging %s", "enabled" if enabled else "disabled")

    def get_verbose_logging(self) -> bool:
        """Get current verbose logging state."""
        return self._verbose_logging

    # Internal helpers -------------------------------------------------------
    def _set_status(self, state: str, details: str, current_track: str = "") -> None:
        with self._status_lock:
            self._status["state"] = state
            self._status["details"] = details
            if current_track:
                self._status["current_track"] = current_track

    def _terminate_process(self) -> None:
        if not self._process:
            return
        logging.info("Stopping spotify-splitter process (pid=%s)", self._process.pid)
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logging.warning("Process did not exit in time; killing")
                self._process.kill()
        finally:
            self._process = None

    def _build_command(self, config: Dict[str, str]) -> list[str]:
        python_exec = sys.executable
        cmd = [
            python_exec,
            "-m",
            "spotify_splitter.main",
        ]
        if self.config_path:
            cmd.extend(["--config", self.config_path])

        record_args: list[str] = []
        metrics_enabled = config.get("enable_metrics")
        monitoring_enabled = config.get("enable_monitoring")

        if metrics_enabled is None:
            metrics_enabled = not self.disable_metrics
        if monitoring_enabled is None:
            monitoring_enabled = not self.disable_metrics

        if not metrics_enabled:
            record_args.append("--no-metrics")

        if not monitoring_enabled:
            record_args.append("--no-monitoring")

        if not config.get("enable_adaptive", True):
            record_args.append("--no-adaptive")
        if config.get("debug_mode"):
            record_args.append("--debug-mode")
        if config.get("player") and config.get("player") != DEFAULT_CONFIG["player"]:
            record_args.extend(["--player", config["player"]])
        if config.get("profile") and config.get("profile") != DEFAULT_CONFIG["profile"]:
            record_args.extend(["--profile", config["profile"]])
        if config.get("playlist"):
            record_args.extend(["--playlist", config["playlist"]])
            if config.get("bundle_playlist"):
                record_args.append("--bundle-playlist")

        cmd.append("record")
        cmd.extend(record_args)
        return cmd

    def _run_loop(self) -> None:
        logging.debug("Supervisor loop started")
        while not self._stop_event.is_set():
            self._restart_event.clear()
            config = load_user_config(self.config_path)
            command = self._build_command(config)
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("RICH_FORCE_TERMINAL", "0")
            env.setdefault("TERM", "dumb")

            log_path = LOG_DIR / "recorder.log"
            stdout_file = log_path.open("a", encoding="utf-8")
            self._set_status("starting", "⏳ Starting recorder...")
            logging.info("Starting spotify-splitter: %s", " ".join(command))

            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            except FileNotFoundError:
                stdout_file.close()
                logging.exception("Failed to launch spotify-splitter")
                self._set_status("error", "spotify-splitter executable not found")
                time.sleep(self.auto_restart_delay)
                continue

            start_time = time.time()

            # Give process a moment to start and verify it's running
            time.sleep(2.5)
            if self._process.poll() is None:
                self._set_status("running", "🎵 Recording - waiting for Spotify playback...")
                logging.info("Recorder started successfully")
            else:
                early_exit = self._process.poll()
                logging.error("Recorder exited immediately with code %s", early_exit)
                stdout_file.close()
                continue

            exit_code = None
            while exit_code is None and not self._restart_event.is_set() and not self._stop_event.is_set():
                exit_code = self._process.poll()
                if exit_code is None:
                    time.sleep(1.0)

            if exit_code is None:
                # Restart requested or stop event set
                self._terminate_process()
                stdout_file.close()
                if self._stop_event.is_set():
                    break
                else:
                    self._set_status("restarting", "Restart requested")
                    continue

            runtime = time.time() - start_time
            stdout_file.close()
            self._process = None

            # Check if manually stopped - don't auto-restart
            if self._manual_stop:
                self._set_status("stopped", "Recording stopped by user")
                break

            if exit_code == 0:
                self._set_status("stopped", "Recorder exited normally")
            else:
                if runtime < 8 and exit_code == 1:
                    message = "Spotify client not ready or no playback detected"
                else:
                    message = (
                        "Recorder exited with code "
                        f"{exit_code} after {runtime:.0f}s"
                    )
                logging.warning(message)
                self._set_status("waiting", message + f"; retrying in {self.auto_restart_delay}s")

                wait_time = self.auto_restart_delay
                while wait_time > 0 and not self._stop_event.is_set() and not self._restart_event.is_set():
                    time.sleep(1.0)
                    wait_time -= 1

        self._terminate_process()
        self._set_status("stopped", "Supervisor loop exited")
        logging.debug("Supervisor loop finished")


class Spoti2RequestHandler(BaseHTTPRequestHandler):
    """Modern 3-tab web UI inspired by spy-spotify design."""

    server_version = "Spoti2Service/2.0"

    # Modern color palette with Spotify green
    palette = {
        "bg": "#0d1117",
        "panel": "#161b22",
        "border": "#30363d",
        "accent": "#1DB954",  # Spotify green
        "accent_hover": "#1ED760",  # Lighter Spotify green for hover
        "text": "#c9d1d9",
        "text_muted": "#8b949e",
        "success": "#1DB954",  # Also use Spotify green for success
        "warning": "#d29922",
        "error": "#f85149",
    }

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/":
            self._serve_index()
        elif self.path == "/status":
            self._send_json(self.server.app.supervisor.status())
        elif self.path == "/logs":
            self._serve_logs()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/update":
            self._handle_update()
        elif self.path == "/restart":
            self._handle_restart()
        elif self.path == "/start":
            self._handle_start()
        elif self.path == "/stop":
            self._handle_stop()
        elif self.path == "/pause":
            self._handle_pause()
        elif self.path == "/resume":
            self._handle_resume()
        elif self.path == "/toggle-verbose":
            self._handle_toggle_verbose()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logging.info("HTTP %s - %s", self.address_string(), format % args)

    # Helpers -----------------------------------------------------------------
    def _serve_index(self) -> None:
        config = load_user_config(self.server.app.config_path)
        status = self.server.app.supervisor.status()
        body = self._render_index(config, status)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _serve_logs(self) -> None:
        """Serve recent filtered log entries and update status based on activity."""
        try:
            log_path = LOG_DIR / "recorder.log"
            if log_path.exists():
                verbose = self.server.app.supervisor.get_verbose_logging()

                with open(log_path, "r") as f:
                    lines = f.readlines()
                    recent_logs = lines[-200:] if len(lines) > 200 else lines

                    # Track change detection for status updates
                    track_change_keywords = ["TRACK CHANGE CALLBACK:", "Track changed:"]
                    last_track = None

                    if verbose:
                        # Verbose mode: show more log events
                        verbose_keywords = [
                            "Saved:",
                            "ERROR:",
                            "WARNING:",
                            "Track changed:",
                            "Starting MPRIS",
                            "already exists",
                            "Recording",
                        ]

                        filtered_lines = []
                        for line in recent_logs:
                            # Extract current track
                            for keyword in track_change_keywords:
                                if keyword in line:
                                    try:
                                        parts = line.split(keyword)
                                        if len(parts) > 1:
                                            last_track = parts[1].strip()
                                    except:
                                        pass

                            # Clean duplicates
                            cleaned = line.strip()
                            cleaned = cleaned.replace("INFO: INFO:", "INFO:")
                            cleaned = cleaned.replace("INFO INFO:", "INFO:")

                            if any(keyword in line for keyword in verbose_keywords):
                                filtered_lines.append(cleaned)

                        logs = "\n".join(filtered_lines[-50:]) if filtered_lines else "🎵 Waiting for tracks to record..."
                    else:
                        # Minimal mode: only show essential status events
                        essential_keywords = [
                            "Saved:",           # Track saved
                            "ERROR:",           # Critical errors only
                            "already exists",   # Duplicate detection
                        ]

                        filtered_lines = []
                        for line in recent_logs:
                            # Extract current track
                            for keyword in track_change_keywords:
                                if keyword in line:
                                    try:
                                        parts = line.split(keyword)
                                        if len(parts) > 1:
                                            last_track = parts[1].strip()
                                    except:
                                        pass

                            # Clean and filter
                            cleaned = line.strip()
                            cleaned = cleaned.replace("INFO: INFO:", "INFO:")
                            cleaned = cleaned.replace("INFO INFO:", "INFO:")

                            if any(keyword in line for keyword in essential_keywords):
                                # Remove INFO/DEBUG prefixes for cleaner display
                                for prefix in ["INFO:", "DEBUG:"]:
                                    if cleaned.startswith(prefix):
                                        cleaned = cleaned[len(prefix):].strip()
                                filtered_lines.append(cleaned)

                        logs = "\n".join(filtered_lines[-30:]) if filtered_lines else "🎵 Waiting for tracks to record..."

                    # Update status if we detected a track change
                    if last_track and self.server.app.supervisor._process:
                        status = self.server.app.supervisor.status()
                        if status.get("state") == "running":
                            self.server.app.supervisor._set_status(
                                "running",
                                f"🎵 Recording: {last_track}"
                            )
            else:
                logs = "No logs available - recorder not started"

            self._send_json({"logs": logs})
        except Exception as e:
            self._send_json({"logs": f"Error reading logs: {e}"})

    def _handle_update(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        form = parse_qs(payload)

        config = load_user_config(self.server.app.config_path)

        def get_value(name: str, default: Optional[str] = None) -> Optional[str]:
            return form.get(name, [default])[0]

        def get_bool(name: str) -> bool:
            return name in form

        updates: Dict[str, object] = {
            "output": get_value("output", config.get("output") or DEFAULT_CONFIG["output"]),
            "format": get_value("format", config.get("format") or "mp3"),
            "player": get_value("player", config.get("player")),
            "profile": get_value("profile", config.get("profile")),
            "playlist": get_value("playlist") or None,
            "bundle_playlist": get_bool("bundle_playlist"),
            "enable_adaptive": get_bool("enable_adaptive"),
            "enable_monitoring": get_bool("enable_monitoring"),
            "enable_metrics": get_bool("enable_metrics"),
            "debug_mode": get_bool("debug_mode"),
            "lastfm_api_key": get_value("lastfm_api_key") or None,
            "allow_overwrite": get_bool("allow_overwrite"),
        }

        merged = config.copy()
        merged.update(updates)
        # Normalize paths
        for key in ("output", "playlist"):
            if merged.get(key):
                merged[key] = str(Path(merged[key]).expanduser())

        save_user_config(merged, self.server.app.config_path)
        self.server.app.supervisor.request_restart("Configuration updated via web UI")

        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _handle_restart(self) -> None:
        self.server.app.supervisor.request_restart("Manual restart from web UI")
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _handle_start(self) -> None:
        """Start recording."""
        self.server.app.supervisor.start()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _handle_stop(self) -> None:
        """Stop recording."""
        self.server.app.supervisor.stop()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _handle_pause(self) -> None:
        """Pause recording."""
        self.server.app.supervisor.pause()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _handle_resume(self) -> None:
        """Resume recording."""
        self.server.app.supervisor.resume()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _handle_toggle_verbose(self) -> None:
        """Toggle verbose logging mode."""
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        form = parse_qs(payload)

        # Checkbox is present if checked, absent if unchecked
        verbose = "verbose" in form
        self.server.app.supervisor.set_verbose_logging(verbose)

        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def _send_json(self, payload: Dict[str, str]) -> None:
        body = json.dumps(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _render_index(self, config: Dict[str, str], status: Dict[str, str]) -> str:
        p = self.palette

        def checked(flag: bool) -> str:
            return "checked" if flag else ""

        state = status.get("state", "unknown")
        state_color = {
            "running": p["success"],
            "starting": p["warning"],
            "stopped": p["text_muted"],
            "paused": p["warning"],
            "waiting": p["warning"],
            "error": p["error"],
        }.get(state, p["text"])

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Spoti2 - Linux Spotify Recorder</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      background: {p["bg"]};
      color: {p["text"]};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.6;
      min-height: 100vh;
    }}

    header {{
      background: {p["panel"]};
      border-bottom: 1px solid {p["border"]};
      padding: 1.5rem 2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}

    header h1 {{
      font-size: 1.75rem;
      font-weight: 600;
      color: {p["accent"]};
      margin-bottom: 0.25rem;
    }}

    header p {{
      color: {p["text_muted"]};
      font-size: 0.9rem;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
    }}

    /* Tab Navigation */
    .tabs {{
      display: flex;
      gap: 0.5rem;
      margin-bottom: 2rem;
      border-bottom: 2px solid {p["border"]};
    }}

    .tab-button {{
      background: none;
      border: none;
      color: {p["text_muted"]};
      padding: 1rem 2rem;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 500;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s;
    }}

    .tab-button:hover {{
      color: {p["text"]};
      background: rgba(255,255,255,0.05);
    }}

    .tab-button.active {{
      color: {p["accent"]};
      border-bottom-color: {p["accent"]};
    }}

    .tab-content {{
      display: none;
    }}

    .tab-content.active {{
      display: block;
    }}

    /* Panels */
    .panel {{
      background: {p["panel"]};
      border: 1px solid {p["border"]};
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}

    .panel h2 {{
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: {p["text"]};
    }}

    /* Status Display */
    .status-display {{
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem;
      background: rgba(29, 185, 84, 0.1);
      border-radius: 6px;
      margin-bottom: 1rem;
    }}

    .status-indicator {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: {state_color};
      animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.5; }}
    }}

    .status-text {{
      flex: 1;
    }}

    .status-text strong {{
      color: {p["accent"]};
      text-transform: capitalize;
    }}

    .status-details {{
      color: {p["text_muted"]};
      font-size: 0.9rem;
    }}

    /* Recording Log */
    .log-container {{
      background: {p["bg"]};
      border: 1px solid {p["border"]};
      border-radius: 6px;
      padding: 1rem;
      max-height: 400px;
      overflow-y: auto;
      font-family: 'Courier New', monospace;
      font-size: 0.85rem;
    }}

    .log-container::-webkit-scrollbar {{
      width: 8px;
    }}

    .log-container::-webkit-scrollbar-track {{
      background: {p["panel"]};
    }}

    .log-container::-webkit-scrollbar-thumb {{
      background: {p["border"]};
      border-radius: 4px;
    }}

    /* Forms */
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}

    label {{
      display: block;
      margin-bottom: 1rem;
    }}

    label span {{
      display: block;
      margin-bottom: 0.5rem;
      color: {p["text"]};
      font-weight: 500;
    }}

    input[type="text"], select {{
      width: 100%;
      padding: 0.75rem;
      background: {p["bg"]};
      border: 1px solid {p["border"]};
      border-radius: 6px;
      color: {p["text"]};
      font-size: 0.95rem;
      transition: border-color 0.2s;
    }}

    input[type="text"]:focus, select:focus {{
      outline: none;
      border-color: {p["accent"]};
    }}

    .checkbox-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      margin: 1rem 0;
    }}

    .checkbox-group label {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 0;
    }}

    input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      cursor: pointer;
    }}

    /* Buttons */
    .button-group {{
      display: flex;
      gap: 1rem;
      margin-top: 1.5rem;
    }}

    button {{
      flex: 1;
      padding: 0.875rem 1.5rem;
      background: linear-gradient(135deg, {p["accent"]}, {p["accent_hover"]});
      border: none;
      border-radius: 6px;
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
    }}

    button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(29, 185, 84, 0.3);
    }}

    button:active {{
      transform: translateY(0);
    }}

    button.secondary {{
      background: {p["panel"]};
      border: 1px solid {p["border"]};
      color: {p["text"]};
    }}

    button.secondary:hover {{
      background: {p["border"]};
      box-shadow: none;
    }}

    .help-text {{
      color: {p["text_muted"]};
      font-size: 0.875rem;
      margin-top: 0.5rem;
    }}

    footer {{
      text-align: center;
      padding: 2rem;
      color: {p["text_muted"]};
      font-size: 0.875rem;
    }}

    footer a {{
      color: {p["accent"]};
      text-decoration: none;
    }}

    footer a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <header>
    <h1>🎵 Spoti2</h1>
    <p>Linux Spotify Desktop Recorder with LastFM Metadata</p>
  </header>

  <div class="container">
    <!-- Tab Navigation -->
    <div class="tabs">
      <button class="tab-button active" onclick="switchTab('record')">Record</button>
      <button class="tab-button" onclick="switchTab('settings')">Settings</button>
      <button class="tab-button" onclick="switchTab('advanced')">Advanced</button>
    </div>

    <!-- Tab 1: Record -->
    <div id="tab-record" class="tab-content active">
      <div class="panel">
        <h2>Recording Status</h2>
        <div class="status-display">
          <div class="status-indicator"></div>
          <div class="status-text">
            <div><strong>{state}</strong></div>
            <div class="status-details">{status.get("details", "")}</div>
          </div>
        </div>

        <div class="button-group">
          <form method="post" action="/start" style="flex: 1;">
            <button type="submit">▶ Start</button>
          </form>
          <form method="post" action="/pause" style="flex: 1;">
            <button type="submit" class="secondary">⏸ Pause</button>
          </form>
          <form method="post" action="/resume" style="flex: 1;">
            <button type="submit" class="secondary">▶▶ Resume</button>
          </form>
          <form method="post" action="/stop" style="flex: 1;">
            <button type="submit" class="secondary">⏹ Stop</button>
          </form>
        </div>
      </div>

      <div class="panel">
        <h2>Recording Log</h2>
        <div class="log-container" id="log-display">
          <div style="color: {p["text_muted"]};">Loading logs...</div>
        </div>

        <form method="post" action="/toggle-verbose" style="margin-top: 1rem;">
          <div class="checkbox-group">
            <label>
              <input type="checkbox" name="verbose" {checked(self.server.app.supervisor.get_verbose_logging())} onchange="this.form.submit()">
              <span style="color: {p["text_muted"]};">Show verbose logs (includes track changes, MPRIS events, warnings)</span>
            </label>
          </div>
        </form>
      </div>
    </div>

    <!-- Tab 2: Settings -->
    <div id="tab-settings" class="tab-content">
      <form class="panel" method="post" action="/update">
        <h2>Output Settings</h2>
        <div class="form-grid">
          <label>
            <span>Output Directory</span>
            <input type="text" name="output" value="{config.get("output", DEFAULT_CONFIG["output"])}" />
            <div class="help-text">Where recorded tracks will be saved</div>
          </label>

          <label>
            <span>Audio Format</span>
            <select name="format">
              {"".join(self._select_options(config.get("format", "mp3"), ["mp3","flac","wav","ogg"]))}
            </select>
            <div class="help-text">Output file format</div>
          </label>
        </div>

        <div class="checkbox-group" style="margin-top: 1rem;">
          <label>
            <input type="checkbox" name="allow_overwrite" {checked(config.get("allow_overwrite", False))}>
            <span>Allow Overwriting Existing Files</span>
          </label>
        </div>
        <div class="help-text" style="margin-top: 0.5rem;">When enabled, tracks will be re-recorded even if they already exist (useful if previous recordings were incomplete)</div>

        <h2 style="margin-top: 2rem;">Metadata Settings</h2>
        <div class="form-grid">
          <label>
            <span>LastFM API Key</span>
            <input type="text" name="lastfm_api_key" value="{config.get("lastfm_api_key", "") or ""}" placeholder="Enter your LastFM API key" />
            <div class="help-text">Required for fetching year and genre tags. <a href="https://www.last.fm/api/account/create" target="_blank" style="color: {p["accent"]};">Get one here</a></div>
          </label>
        </div>

        <h2 style="margin-top: 2rem;">Playlist Settings</h2>
        <div class="form-grid">
          <label>
            <span>Playlist File (optional)</span>
            <input type="text" name="playlist" value="{config.get("playlist", "") or ""}" placeholder="/path/to/playlist.m3u" />
            <div class="help-text">Generate an M3U playlist file</div>
          </label>
        </div>

        <div class="checkbox-group">
          <label>
            <input type="checkbox" name="bundle_playlist" {checked(config.get("bundle_playlist", False))}>
            <span>Bundle as Compilation Album</span>
          </label>
        </div>

        <div class="button-group">
          <button type="submit">Save Settings</button>
        </div>
      </form>
    </div>

    <!-- Tab 3: Advanced -->
    <div id="tab-advanced" class="tab-content">
      <form class="panel" method="post" action="/update">
        <h2>Performance Settings</h2>

        <label>
          <span>Configuration Profile</span>
          <select name="profile">
            {"".join(self._select_options(config.get("profile", "auto"), ["auto","desktop","headless","high_performance"]))}
          </select>
          <div class="help-text">Optimize for your system</div>
        </label>

        <div class="checkbox-group" style="margin-top: 1.5rem;">
          <label>
            <input type="checkbox" name="enable_adaptive" {checked(config.get("enable_adaptive", True))}>
            <span>Adaptive Buffers</span>
          </label>
          <label>
            <input type="checkbox" name="enable_monitoring" {checked(config.get("enable_monitoring", False))}>
            <span>Buffer Monitoring</span>
          </label>
          <label>
            <input type="checkbox" name="enable_metrics" {checked(config.get("enable_metrics", False))}>
            <span>Performance Metrics</span>
          </label>
          <label>
            <input type="checkbox" name="debug_mode" {checked(config.get("debug_mode", False))}>
            <span>Debug Mode</span>
          </label>
        </div>

        <h2 style="margin-top: 2rem;">Player Settings</h2>

        <label>
          <span>MPRIS Player Name</span>
          <input type="text" name="player" value="{config.get("player", DEFAULT_CONFIG["player"])}" />
          <div class="help-text">Usually "spotify" for Spotify desktop client</div>
        </label>

        <div class="button-group">
          <button type="submit">Save Advanced Settings</button>
        </div>
      </form>
    </div>
  </div>

  <footer>
    <p>Spoti2 &mdash; Linux Spotify Recorder | <a href="https://github.com/Brownster/spytify-LE#readme" target="_blank">Documentation</a></p>
  </footer>

  <script>
    function switchTab(tabName) {{
      // Hide all tabs
      document.querySelectorAll('.tab-content').forEach(tab => {{
        tab.classList.remove('active');
      }});
      document.querySelectorAll('.tab-button').forEach(btn => {{
        btn.classList.remove('active');
      }});

      // Show selected tab
      document.getElementById('tab-' + tabName).classList.add('active');
      event.target.classList.add('active');
    }}

    // Auto-refresh logs every 3 seconds
    function refreshLogs() {{
      fetch('/logs')
        .then(response => response.json())
        .then(data => {{
          const logDisplay = document.getElementById('log-display');
          if (data.logs) {{
            logDisplay.textContent = data.logs;
            logDisplay.scrollTop = logDisplay.scrollHeight;
          }}
        }})
        .catch(err => console.error('Failed to fetch logs:', err));
    }}

    // Refresh status every 2 seconds
    function refreshStatus() {{
      fetch('/status')
        .then(response => response.json())
        .then(data => {{
          // Update status in UI (could enhance this later)
          console.log('Status:', data);
        }})
        .catch(err => console.error('Failed to fetch status:', err));
    }}

    // Initial load and set intervals
    refreshLogs();
    setInterval(refreshLogs, 3000);
    setInterval(refreshStatus, 2000);
  </script>
</body>
</html>
        """.strip()

    @staticmethod
    def _select_options(current: str, options: list[str]) -> list[str]:
        rendered = []
        for option in options:
            selected = "selected" if option == (current or options[0]) else ""
            rendered.append(f'<option value="{option}" {selected}>{option.title()}</option>')
        return rendered


class Spoti2Service:
    """Coordinates HTTP UI and recorder supervisor."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8730,
        config_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        configure_logging(verbose)
        self.host = host
        self.port = port
        self.config_path = config_path
        self.supervisor = RecorderSupervisor(config_path=config_path)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        logging.info("Starting Spoti2Service (UI on http://%s:%d)", self.host, self.port)
        self._setup_http_server()
        self.supervisor.start()
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()
        self._install_signal_handlers()
        self._wait_forever()

    def _setup_http_server(self) -> None:
        config_path = self.config_path or str(get_config_path())
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(config_path).exists():
            logging.info("Initializing configuration file at %s", config_path)
            save_user_config(DEFAULT_CONFIG.copy(), self.config_path)

        server = ThreadingHTTPServer((self.host, self.port), Spoti2RequestHandler)
        server.app = self
        self._httpd = server

    def _install_signal_handlers(self) -> None:
        def handle_shutdown(signum, frame):
            logging.info("Received signal %s, shutting down service", signum)
            self.stop()

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

    def _wait_forever(self) -> None:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received; stopping service")
            self.stop()

    def stop(self) -> None:
        if self._httpd:
            logging.info("Stopping HTTP server")
            self._httpd.shutdown()
            self._httpd.server_close()
        self.supervisor.stop()


def run_service(host: str = "0.0.0.0", port: int = 8730, config: Optional[str] = None, verbose: bool = False) -> None:
    service = Spoti2Service(host=host, port=port, config_path=config, verbose=verbose)
    service.start()
