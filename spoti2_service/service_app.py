import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from spotify_splitter.metadata_edit import (
    MetadataEditError,
    edit_track_metadata,
    validate_year,
)
from spotify_splitter.doctor import run_doctor
from spotify_splitter.track_history import TrackHistoryWriter
from spotify_splitter.user_config import (
    DEFAULT_CONFIG,
    get_config_path,
    load_user_config,
    save_user_config,
)

from .web_ui import PALETTE, render_index


LOG_DIR = Path.home() / ".cache" / "spotify_splitter" / "service"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "service.log"
RECORDER_LOG_PATH = LOG_DIR / "recorder.log"
RECORDER_STATUS_PATH = LOG_DIR / "status.json"
RECORDER_HISTORY_PATH = LOG_DIR / "history.jsonl"
RECORDER_STATUS_STALE_SECONDS = 15.0


def merge_web_config(config: Dict[str, Any], form: Dict[str, list]) -> Dict[str, Any]:
    """Merge a posted web-UI form into ``config``, updating only present fields.

    A form only carries the fields it owns, so partial forms (e.g. the timer-only
    form, or a single toggle) must not wipe settings they omit. Checkboxes use a
    hidden companion input of the same name, so an unchecked box still appears in
    the form as a falsy value and can be turned off.
    """
    merged = dict(config)

    # Empty string clears these to None; for the rest, empty falls back to existing.
    nullable_text = {
        "playlist", "bundle_album_art_uri", "playlist_base_path",
        "max_duration", "lastfm_api_key",
    }
    text_fields = [
        "output", "format", "player", "profile",
        "playlist", "bundle_album_art_uri", "playlist_base_path",
        "max_duration", "lastfm_api_key",
    ]
    for key in text_fields:
        if key in form:
            value = form[key][0].strip()
            merged[key] = (value or None) if key in nullable_text else (value or merged.get(key))

    bool_fields = ["bundle_playlist", "allow_overwrite"]
    for key in bool_fields:
        if key in form:
            merged[key] = form[key][-1] not in ("", "0", "off", "false")

    for key in ("output", "playlist", "playlist_base_path"):
        if merged.get(key):
            merged[key] = str(Path(merged[key]).expanduser())

    return merged


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
        status_path: Path = RECORDER_STATUS_PATH,
        graceful_stop_timeout: float = 20.0,
    ) -> None:
        self.config_path = config_path
        self.auto_restart_delay = auto_restart_delay
        self.status_path = status_path
        self.graceful_stop_timeout = graceful_stop_timeout

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
            "timer_enabled": False,
            "timer_start_time": 0,
            "timer_duration_seconds": 0,
        }

    # Public API -------------------------------------------------------------
    def start(self) -> None:
        """Start or resume recording."""
        self._manual_stop = False
        if self._thread and self._thread.is_alive():
            if self._pause_event.is_set():
                # Resume from pause
                logging.info("Resuming recording")
                self.resume()
            else:
                # Restart if stopped
                logging.info("Restarting recording")
                self._restart_event.set()
            return

        logging.info("Starting RecorderSupervisor thread")
        # Fresh session: clear the previous run's log so stale errors from an
        # earlier recorder can't masquerade as current activity in the UI.
        self._truncate_recorder_log()
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _truncate_recorder_log(self) -> None:
        try:
            RECORDER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            RECORDER_LOG_PATH.write_text("", encoding="utf-8")
        except Exception as e:
            logging.debug("Could not truncate recorder log: %s", e)

    def _recorder_log_reason(self) -> str:
        """Best-effort one-line reason from the tail of the recorder log."""
        try:
            lines = [
                ln.strip()
                for ln in RECORDER_LOG_PATH.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        except Exception:
            return ""
        for line in reversed(lines[-25:]):
            if any(tok in line for tok in ("Error", "Exception", "Traceback", "No module named")):
                return line[:200]
        return lines[-1][:200] if lines else ""

    def stop(self) -> None:
        """Stop recording (user-initiated)."""
        logging.info("Stopping RecorderSupervisor (manual stop)")
        self._manual_stop = True
        if self._pause_event.is_set():
            self.resume()
        self._pause_event.clear()
        proc = self._process
        if proc and proc.poll() is None and self._request_graceful_stop():
            self._set_status("stopped", "⏹️ Recording stopped")
            return
        self._stop_event.set()
        self._restart_event.set()
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

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            supervisor_status = dict(self._status)
        return self._merge_recorder_status(supervisor_status)

    # Internal helpers -------------------------------------------------------
    def _set_status(self, state: str, details: str, current_track: str = "") -> None:
        with self._status_lock:
            self._status["state"] = state
            self._status["details"] = details
            if current_track:
                self._status["current_track"] = current_track

    def _read_recorder_status(self) -> Optional[Dict[str, Any]]:
        try:
            with self.status_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            logging.debug("Failed to read recorder status file: %s", e)
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _is_recorder_status_current(self, recorder_status: Dict[str, Any]) -> bool:
        proc = self._process
        if not proc or proc.poll() is not None:
            return False

        pid = recorder_status.get("pid")
        if pid and pid != proc.pid:
            return False

        # A paused process cannot update the status file while SIGSTOP'd, so pid
        # is the freshness check for paused state.
        if self._pause_event.is_set():
            return True

        updated_at = recorder_status.get("updated_at")
        if not updated_at:
            return True
        try:
            timestamp = updated_at.replace("Z", "+00:00")
            updated = datetime.fromisoformat(timestamp)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
        except Exception:
            return True
        return (datetime.now(timezone.utc) - updated).total_seconds() <= RECORDER_STATUS_STALE_SECONDS

    def _merge_recorder_status(self, supervisor_status: Dict[str, Any]) -> Dict[str, Any]:
        recorder_status = self._read_recorder_status()
        if not recorder_status:
            return supervisor_status

        merged = dict(supervisor_status)
        is_current = self._is_recorder_status_current(recorder_status)
        merged["recorder_status_stale"] = not is_current
        if not is_current:
            return merged

        current_track = recorder_status.get("current_track")
        if isinstance(current_track, dict):
            # Full track detail for the now-playing card (art, duration, position).
            merged["track"] = current_track
            artist = current_track.get("artist") or ""
            title = current_track.get("title") or ""
            track_label = " - ".join(part for part in [artist, title] if part)
            if track_label:
                merged["current_track"] = track_label
                if merged.get("state") == "running":
                    merged["details"] = f"🎵 Recording: {track_label}"

        merged["tracks_recorded"] = recorder_status.get("tracks_recorded", 0)
        merged["recorder_state"] = recorder_status.get("state")
        merged["last_error"] = recorder_status.get("last_error")
        merged["updated_at"] = recorder_status.get("updated_at")
        merged["samplerate"] = recorder_status.get("samplerate", 0)
        merged["output_format"] = recorder_status.get("output_format", "")

        timer = recorder_status.get("timer")
        if isinstance(timer, dict):
            merged["timer"] = timer
            merged["timer_enabled"] = bool(timer.get("enabled"))
            merged["timer_elapsed_seconds"] = int(timer.get("elapsed_seconds") or 0)
            merged["timer_remaining_seconds"] = int(timer.get("remaining_seconds") or 0)

        audio = recorder_status.get("audio")
        if isinstance(audio, dict):
            merged["audio"] = audio
            merged["queue_depth"] = int(audio.get("queue_depth") or 0)
            merged["dropped_frames"] = int(audio.get("dropped_frames") or 0)
            merged["buffer_warnings"] = int(audio.get("buffer_warnings") or 0)

        return merged

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

    def _send_control_command(self, command: Dict[str, Any]) -> bool:
        proc = self._process
        if not proc or proc.poll() is not None or not proc.stdin:
            return False
        try:
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()
            return True
        except Exception as e:
            logging.warning("Failed to send recorder control command: %s", e)
            return False

    def _request_graceful_stop(self) -> bool:
        proc = self._process
        if not proc or proc.poll() is not None:
            return True

        logging.info("Requesting graceful recorder stop (pid=%s)", proc.pid)
        self._set_status("stopping", "Finalizing recording...")
        if not self._send_control_command({"cmd": "stop", "flush": True}):
            return False
        try:
            proc.wait(timeout=self.graceful_stop_timeout)
            logging.info("Recorder exited after graceful stop request")
            return True
        except subprocess.TimeoutExpired:
            logging.warning(
                "Recorder did not exit within %.1fs after graceful stop request",
                self.graceful_stop_timeout,
            )
            return False

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
        if config.get("player") and config.get("player") != DEFAULT_CONFIG["player"]:
            record_args.extend(["--player", config["player"]])
        if config.get("profile") and config.get("profile") != DEFAULT_CONFIG["profile"]:
            record_args.extend(["--profile", config["profile"]])
        if config.get("playlist"):
            record_args.extend(["--playlist", config["playlist"]])
            if config.get("bundle_playlist"):
                record_args.append("--bundle-playlist")
            if config.get("bundle_album_art_uri"):
                record_args.extend(["--bundle-album-art-uri", config["bundle_album_art_uri"]])
            if config.get("playlist_base_path"):
                record_args.extend(["--playlist-base-path", config["playlist_base_path"]])
        if config.get("max_duration"):
            record_args.extend(["--max-duration", config["max_duration"]])
        record_args.extend(["--status-file", str(self.status_path)])
        record_args.append("--control-stdin")
        record_args.extend(["--history-file", str(RECORDER_HISTORY_PATH)])

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

            log_path = RECORDER_LOG_PATH
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.unlink(missing_ok=True)
            stdout_file = log_path.open("a", encoding="utf-8")
            self._set_status("starting", "⏳ Starting recorder...")
            logging.info("Starting spotify-splitter: %s", " ".join(command))

            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
            except FileNotFoundError:
                stdout_file.close()
                logging.exception("Failed to launch spotify-splitter")
                self._set_status("error", "spotify-splitter executable not found")
                time.sleep(self.auto_restart_delay)
                continue

            start_time = time.time()

            # Update timer status if max_duration is configured
            if config.get("max_duration"):
                from spotify_splitter.duration_parser import parse_duration
                try:
                    timer_duration = parse_duration(config["max_duration"])
                    with self._status_lock:
                        self._status["timer_enabled"] = True
                        self._status["timer_start_time"] = start_time
                        self._status["timer_duration_seconds"] = timer_duration
                except ValueError:
                    logging.warning(f"Invalid max_duration format: {config['max_duration']}")

            # Give process a moment to start and verify it's running
            time.sleep(2.5)
            if self._process.poll() is None:
                self._set_status("running", "🎵 Recording - waiting for Spotify playback...")
                logging.info("Recorder started successfully")
            else:
                early_exit = self._process.poll()
                stdout_file.close()
                reason = self._recorder_log_reason()
                logging.error("Recorder exited immediately with code %s: %s", early_exit, reason)
                detail = f"❌ Recorder failed to start (exit {early_exit})"
                if reason:
                    detail += f": {reason}"
                self._set_status("error", detail)
                if self._manual_stop or self._stop_event.is_set():
                    break
                # Back off before respawning so a crash-loop doesn't spin.
                time.sleep(self.auto_restart_delay)
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

            # Reset timer status
            with self._status_lock:
                self._status["timer_enabled"] = False
                self._status["timer_start_time"] = 0
                self._status["timer_duration_seconds"] = 0

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
        with self._status_lock:
            if self._status.get("state") != "stopped":
                self._status["state"] = "stopped"
                self._status["details"] = "Supervisor loop exited"
        logging.debug("Supervisor loop finished")


class Spoti2RequestHandler(BaseHTTPRequestHandler):
    """Modern 3-tab web UI inspired by spy-spotify design."""

    server_version = "Spoti2Service/2.0"

    palette = PALETTE

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/":
            self._serve_index()
        elif self.path == "/status":
            self._send_json(self.server.app.supervisor.status())
        elif self.path == "/history":
            self._serve_history()
        elif self.path == "/doctor":
            self._serve_doctor()
        elif self.path == "/logo.png":
            self._serve_logo()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_history(self) -> None:
        """Serve recent per-track recording outcomes, newest first."""
        try:
            records = TrackHistoryWriter(RECORDER_HISTORY_PATH).read(limit=200)
        except Exception as e:
            logging.debug("Failed to read track history: %s", e)
            records = []
        self._send_json({"records": records})

    def _serve_doctor(self) -> None:
        """Serve current first-run diagnostics for the web UI."""
        try:
            report = run_doctor(self.server.app.config_path)
            payload = report.to_dict()
        except Exception as e:  # pragma: no cover - defensive
            logging.debug("Failed to run doctor checks: %s", e)
            payload = {
                "ok": False,
                "summary": "Could not run checks",
                "checks": [{
                    "id": "doctor",
                    "label": "Diagnostics",
                    "status": "error",
                    "message": str(e),
                    "action": "Check service logs for details.",
                }],
            }
        self._send_json(payload)

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
        elif self.path == "/edit-metadata":
            self._handle_edit_metadata()
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

    def _serve_logo(self) -> None:
        """Serve the logo.png file."""
        try:
            # Get the logo path relative to this file
            logo_path = Path(__file__).parent.parent / "logo.png"

            if not logo_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            with open(logo_path, "rb") as f:
                logo_data = f.read()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(logo_data)))
            self.send_header("Cache-Control", "public, max-age=86400")  # Cache for 1 day
            self.end_headers()
            self.wfile.write(logo_data)
        except Exception as e:
            logging.error(f"Error serving logo: {e}")
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_edit_metadata(self) -> None:
        """Rewrite a saved track's year/genre tags and sync the history record."""
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        path = form.get("path", [""])[0]
        genre = (form.get("genre", [""])[0] or "").strip() or None
        config = load_user_config(self.server.app.config_path)
        output_dir = config.get("output") or DEFAULT_CONFIG["output"]
        try:
            year = validate_year(form.get("year", [""])[0])
            edit_track_metadata(path, output_dir, year, genre)
            TrackHistoryWriter(RECORDER_HISTORY_PATH).update_metadata(path, year, genre)
        except MetadataEditError as e:
            self._send_json({"ok": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as e:  # pragma: no cover - defensive
            logging.error("Metadata edit failed: %s", e)
            self._send_json({"ok": False, "error": "edit failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"ok": True})

    def _handle_update(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        form = parse_qs(payload)

        config = load_user_config(self.server.app.config_path)
        merged = merge_web_config(config, form)

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

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _render_index(self, config: Dict[str, str], status: Dict[str, str]) -> str:
        return render_index(config=config, status=status)



class Spoti2Service:
    """Coordinates HTTP UI and recorder supervisor."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8730,
        config_path: Optional[str] = None,
        verbose: bool = False,
        open_browser: bool = False,
    ) -> None:
        configure_logging(verbose)
        self.host = host
        self.port = port
        self.config_path = config_path
        self.open_browser = open_browser
        self.supervisor = RecorderSupervisor(config_path=config_path)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        logging.info("Starting Spoti2Service (UI on http://%s:%d)", self.host, self.port)
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            logging.warning(
                "Web UI is bound to %s without authentication; anyone who can reach it can control recording",
                self.host,
            )
        self._setup_http_server()
        if self.open_browser:
            webbrowser.open(f"http://{self.host}:{self.port}")
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


def run_service(
    host: str = "127.0.0.1",
    port: int = 8730,
    config: Optional[str] = None,
    verbose: bool = False,
    open_browser: bool = False,
) -> None:
    service = Spoti2Service(
        host=host,
        port=port,
        config_path=config,
        verbose=verbose,
        open_browser=open_browser,
    )
    service.start()
