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
        self._status_lock = threading.Lock()
        self._status: Dict[str, str] = {
            "state": "stopped",
            "details": "Service not started",
            "last_exit": "",
        }

    # Public API -------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.debug("RecorderSupervisor already running")
            return
        logging.info("Starting RecorderSupervisor thread")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logging.info("Stopping RecorderSupervisor")
        self._stop_event.set()
        self._restart_event.set()
        self._terminate_process()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._set_status("stopped", "Service stopped by user")

    def request_restart(self, reason: str = "Configuration updated") -> None:
        logging.info("Restart requested: %s", reason)
        self._restart_event.set()
        self._terminate_process()
        self._set_status("restarting", reason)

    def status(self) -> Dict[str, str]:
        with self._status_lock:
            return dict(self._status)

    # Internal helpers -------------------------------------------------------
    def _set_status(self, state: str, details: str) -> None:
        with self._status_lock:
            self._status["state"] = state
            self._status["details"] = details

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
        if config.get("spotifyd_mode"):
            record_args.append("--spotifyd-mode")
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
            self._set_status("starting", "Launching recorder")
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
    """Simple retro-styled configuration UI."""

    server_version = "Spoti2Service/0.1"
    winmx_palette = {
        "background": "#0d1321",
        "panel": "#111d34",
        "accent": "#32c8ff",
        "accent_dark": "#1f9ad6",
        "text": "#d9f0ff",
    }

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path not in ("/", "/status"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if self.path == "/status":
            self._send_json(self.server.app.supervisor.status())
            return
        config = load_user_config(self.server.app.config_path)
        status = self.server.app.supervisor.status()
        body = self._render_index(config, status)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/update":
            self._handle_update()
        elif self.path == "/restart":
            self._handle_restart()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logging.info("HTTP %s - %s", self.address_string(), format % args)

    # Helpers -----------------------------------------------------------------
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
            "spotifyd_mode": get_bool("spotifyd_mode"),
            "enable_adaptive": get_bool("enable_adaptive"),
            "enable_monitoring": get_bool("enable_monitoring"),
            "enable_metrics": get_bool("enable_metrics"),
            "debug_mode": get_bool("debug_mode"),
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

    def _send_json(self, payload: Dict[str, str]) -> None:
        body = json.dumps(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _render_index(self, config: Dict[str, str], status: Dict[str, str]) -> str:
        palette = self.winmx_palette
        def checked(flag: bool) -> str:
            return "checked" if flag else ""

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Spoti2 Control Deck</title>
  <style>
    body {{
      background: radial-gradient(circle at top, {palette["background"]}, #000);
      color: {palette["text"]};
      font-family: "Trebuchet MS", Tahoma, sans-serif;
      margin: 0;
      padding: 0;
    }}
    header {{
      padding: 20px;
      text-align: center;
      background: linear-gradient(90deg, {palette["accent_dark"]}, {palette["accent"]});
      box-shadow: 0 0 20px #000;
    }}
    header h1 {{
      margin: 0;
      letter-spacing: 3px;
      text-transform: uppercase;
    }}
    .container {{
      max-width: 960px;
      margin: 30px auto 40px;
      padding: 0 20px;
    }}
    .panel {{
      background: {palette["panel"]};
      border: 1px solid {palette["accent_dark"]};
      border-radius: 10px;
      box-shadow: 0 0 15px rgba(0,0,0,0.6);
      padding: 20px 30px;
      margin-bottom: 25px;
    }}
    .panel h2 {{
      margin-top: 0;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: {palette["accent"]};
    }}
    label {{
      display: block;
      margin-bottom: 10px;
      font-size: 0.9rem;
    }}
    input[type="text"], select {{
      width: 100%;
      padding: 8px 10px;
      border: 1px solid {palette["accent_dark"]};
      border-radius: 6px;
      background: rgba(0,0,0,0.35);
      color: {palette["text"]};
      margin-top: 4px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 20px;
    }}
    .checkboxes label {{
      display: inline-flex;
      align-items: center;
      margin-right: 20px;
    }}
    .buttons {{
      display: flex;
      gap: 15px;
      margin-top: 20px;
    }}
    button {{
      flex: 1;
      padding: 12px;
      border: none;
      border-radius: 999px;
      background: linear-gradient(90deg, {palette["accent_dark"]}, {palette["accent"]});
      color: #001219;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 2px;
      cursor: pointer;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5);
      transition: transform 0.1s ease, box-shadow 0.1s ease;
    }}
    button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }}
    .status {{
      font-size: 1.1rem;
      margin-bottom: 10px;
    }}
    .status span {{
      color: {palette["accent"]};
    }}
    footer {{
      text-align: center;
      color: rgba(255,255,255,0.55);
      padding-bottom: 30px;
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Spoti2 Control Deck</h1>
    <p>Retro vibes for your recording rig</p>
  </header>
  <div class="container">
    <section class="panel">
      <h2>Status Monitor</h2>
      <div class="status">State: <span>{status.get("state", "unknown").title()}</span></div>
      <div>{status.get("details", "")}</div>
    </section>

    <form class="panel" method="post" action="/update">
      <h2>Session Settings</h2>
      <div class="grid">
        <label>
          Output Directory
          <input type="text" name="output" value="{config.get("output", DEFAULT_CONFIG["output"])}" />
        </label>
        <label>
          Playlist File (optional)
          <input type="text" name="playlist" value="{config.get("playlist", "") or ""}" />
        </label>
        <label>
          Audio Format
          <select name="format">
            {"".join(self._select_options(config.get("format", "mp3"), ["mp3","flac","wav","ogg"]))}
          </select>
        </label>
        <label>
          Player Name
          <input type="text" name="player" value="{config.get("player", DEFAULT_CONFIG["player"])}" />
        </label>
        <label>
          Profile
          <select name="profile">
            {"".join(self._select_options(config.get("profile", "auto"), ["auto","desktop","headless","high_performance"]))}
          </select>
        </label>
      </div>
      <div class="checkboxes" style="margin-top: 20px;">
        <label><input type="checkbox" name="bundle_playlist" {checked(config.get("bundle_playlist", False))}> Bundle Playlist</label>
        <label><input type="checkbox" name="spotifyd_mode" {checked(config.get("spotifyd_mode", False))}> Spotifyd Mode</label>
        <label><input type="checkbox" name="enable_adaptive" {checked(config.get("enable_adaptive", True))}> Adaptive Buffers</label>
        <label><input type="checkbox" name="enable_monitoring" {checked(config.get("enable_monitoring", False))}> Buffer Monitoring</label>
        <label><input type="checkbox" name="enable_metrics" {checked(config.get("enable_metrics", False))}> Metrics</label>
        <label><input type="checkbox" name="debug_mode" {checked(config.get("debug_mode", False))}> Debug Dashboard</label>
      </div>
      <div class="buttons">
        <button type="submit">Save Changes</button>
      </div>
    </form>

    <form class="panel" method="post" action="/restart">
      <h2>Control</h2>
      <p>Need to apply changes or unstick the recorder? Give it a gentle nudge.</p>
      <div class="buttons">
        <button type="submit">Restart Recorder</button>
      </div>
    </form>
  </div>
  <footer>Spoti2 Service &mdash; built for hands-off recording sessions</footer>
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
