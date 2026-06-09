"""HTML rendering for the Spoti2 web UI."""

from html import escape
from typing import Any, Dict

from spotify_splitter.user_config import DEFAULT_CONFIG


PALETTE = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "border": "#30363d",
    "accent": "#1DB954",
    "accent_hover": "#1ED760",
    "text": "#c9d1d9",
    "text_muted": "#8b949e",
    "success": "#1DB954",
    "warning": "#d29922",
    "error": "#f85149",
}


def _select_options(current: str, options: list[str]) -> list[str]:
    rendered = []
    for option in options:
        selected = "selected" if option == (current or options[0]) else ""
        rendered.append(f'<option value="{option}" {selected}>{option.title()}</option>')
    return rendered


def render_index(
    config: Dict[str, Any],
    status: Dict[str, Any],
    verbose_logging: bool,
) -> str:
    """Render the main web UI page."""
    p = PALETTE

    def checked(flag: bool) -> str:
        return "checked" if flag else ""

    def attr(value: object) -> str:
        return escape(str(value or ""), quote=True)

    def text(value: object) -> str:
        return escape(str(value or ""))

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

    header .logo {{
      height: 60px;
      margin-bottom: 0.5rem;
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

    /* Log Entry Styles */
    .log-line, .log-success, .log-error, .log-warning, .log-info, .log-track, .log-waiting {{
      padding: 0.5rem 0.75rem;
      margin-bottom: 0.5rem;
      border-left: 3px solid transparent;
      border-radius: 4px;
      line-height: 1.5;
    }}

    .log-success {{
      background: rgba(29, 185, 84, 0.1);
      border-left-color: {p["success"]};
      color: {p["success"]};
    }}

    .log-error {{
      background: rgba(248, 81, 73, 0.1);
      border-left-color: {p["error"]};
      color: {p["error"]};
    }}

    .log-warning {{
      background: rgba(210, 153, 34, 0.1);
      border-left-color: {p["warning"]};
      color: {p["warning"]};
    }}

    .log-info {{
      background: rgba(29, 185, 84, 0.05);
      border-left-color: {p["accent"]};
      color: {p["text"]};
    }}

    .log-track {{
      background: rgba(29, 185, 84, 0.08);
      border-left-color: {p["accent"]};
      color: {p["accent"]};
      font-weight: 500;
    }}

    .log-line {{
      background: rgba(255, 255, 255, 0.02);
      border-left-color: {p["border"]};
      color: {p["text_muted"]};
    }}

    .log-waiting {{
      color: {p["text_muted"]};
      text-align: center;
      padding: 2rem;
      font-style: italic;
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
      accent-color: {p["accent"]};
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
    <img src="/logo.png" alt="Spytify-LE" class="logo" />
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
          <div id="status-indicator" class="status-indicator"></div>
          <div class="status-text">
            <div><strong id="status-state">{state}</strong></div>
            <div id="status-details" class="status-details">{text(status.get("details", ""))}</div>
            <div id="timer-display" class="status-details" style="display: none; margin-top: 0.5rem; font-weight: 600;"></div>
          </div>
        </div>

        <form method="post" action="/update" style="margin-bottom: 1rem;">
          <label style="display: block; margin-bottom: 0.5rem;">
            <span style="color: {p["text"]}; font-weight: 500;">⏱️ Recording Timer (Optional)</span>
            <input type="text" name="max_duration" value="{attr(config.get("max_duration", ""))}"
                   placeholder="e.g., 4h29m, 90m, 2h30m"
                   style="width: 100%; padding: 0.75rem; margin-top: 0.5rem; background: {p["bg"]}; border: 1px solid {p["border"]}; border-radius: 6px; color: {p["text"]}; font-size: 0.95rem;" />
            <div class="help-text">Automatically stop recording after specified duration. Leave empty for continuous recording.</div>
          </label>
          <button type="submit" class="secondary" style="width: auto; padding: 0.5rem 1rem; font-size: 0.9rem;">Save Timer</button>
        </form>

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
              <input type="checkbox" name="verbose" {checked(verbose_logging)} onchange="this.form.submit()">
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
            <input type="text" name="output" value="{attr(config.get("output", DEFAULT_CONFIG["output"]))}" />
            <div class="help-text">Where recorded tracks will be saved</div>
          </label>

          <label>
            <span>Audio Format</span>
            <select name="format">
              {"".join(_select_options(config.get("format", "mp3"), ["mp3","flac","wav","ogg"]))}
            </select>
            <div class="help-text">Output file format</div>
          </label>
        </div>

        <div class="checkbox-group" style="margin-top: 1rem;">
          <label>
            <input type="hidden" name="allow_overwrite" value="0">
            <input type="checkbox" name="allow_overwrite" value="on" {checked(config.get("allow_overwrite", False))}>
            <span>Allow Overwriting Existing Files</span>
          </label>
        </div>
        <div class="help-text" style="margin-top: 0.5rem;">When enabled, tracks will be re-recorded even if they already exist (useful if previous recordings were incomplete)</div>

        <h2 style="margin-top: 2rem;">Metadata Settings</h2>
        <div class="form-grid">
          <label>
            <span>LastFM API Key</span>
            <input type="text" name="lastfm_api_key" value="{attr(config.get("lastfm_api_key", "") or "")}" placeholder="Enter your LastFM API key" />
            <div class="help-text">Required for fetching year and genre tags. <a href="https://www.last.fm/api/account/create" target="_blank" style="color: {p["accent"]};">Get one here</a></div>
          </label>
        </div>

        <h2 style="margin-top: 2rem;">Playlist Settings</h2>
        <div class="form-grid">
          <label>
            <span>Playlist File (optional)</span>
            <input type="text" name="playlist" value="{attr(config.get("playlist", "") or "")}" placeholder="/path/to/playlist.m3u" />
            <div class="help-text">Generate an M3U playlist file</div>
          </label>
        </div>

        <div class="checkbox-group">
          <label>
            <input type="hidden" name="bundle_playlist" value="0">
            <input type="checkbox" name="bundle_playlist" value="on" {checked(config.get("bundle_playlist", False))}>
            <span>Bundle as Compilation Album</span>
          </label>
        </div>

        <div class="form-group">
          <label>
            <span>Bundle Album Artwork URL (optional)</span>
            <input type="text" name="bundle_album_art_uri" value="{attr(config.get("bundle_album_art_uri", "") or "")}" placeholder="https://example.com/album-cover.jpg" />
            <div class="help-text">Custom album artwork for bundle playlists (uses first track's artwork if not provided)</div>
          </label>
        </div>

        <div class="form-group">
          <label>
            <span>M3U Playlist Base Path (optional)</span>
            <input type="text" name="playlist_base_path" value="{attr(config.get("playlist_base_path", "") or "")}" placeholder="/mnt/storage/music" />
            <div class="help-text">Base path for M3U entries. Maps local recording paths to remote server paths (e.g., recording to ~/Music but listing as /mnt/nas/music in playlist)</div>
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
            {"".join(_select_options(config.get("profile", "auto"), ["auto","desktop","headless","high_performance"]))}
          </select>
          <div class="help-text">Optimize for your system</div>
        </label>

        <div class="checkbox-group" style="margin-top: 1.5rem;">
          <label>
            <input type="hidden" name="enable_adaptive" value="0">
            <input type="checkbox" name="enable_adaptive" value="on" {checked(config.get("enable_adaptive", True))}>
            <span>Adaptive Buffers</span>
          </label>
          <label>
            <input type="hidden" name="enable_monitoring" value="0">
            <input type="checkbox" name="enable_monitoring" value="on" {checked(config.get("enable_monitoring", False))}>
            <span>Buffer Monitoring</span>
          </label>
          <label>
            <input type="hidden" name="enable_metrics" value="0">
            <input type="checkbox" name="enable_metrics" value="on" {checked(config.get("enable_metrics", False))}>
            <span>Performance Metrics</span>
          </label>
          <label>
            <input type="hidden" name="debug_mode" value="0">
            <input type="checkbox" name="debug_mode" value="on" {checked(config.get("debug_mode", False))}>
            <span>Debug Mode</span>
          </label>
        </div>

        <h2 style="margin-top: 2rem;">Player Settings</h2>

        <label>
          <span>MPRIS Player Name</span>
          <input type="text" name="player" value="{attr(config.get("player", DEFAULT_CONFIG["player"]))}" />
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
            logDisplay.innerHTML = data.logs;
            // Keep scroll at top since newest entries are at top
            logDisplay.scrollTop = 0;
          }}
        }})
        .catch(err => console.error('Failed to fetch logs:', err));
    }}

    // Format seconds as "1h 2m 3s"
    function formatTime(seconds) {{
      if (seconds <= 0) return "0s";
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const secs = seconds % 60;
      const parts = [];
      if (hours > 0) parts.push(hours + "h");
      if (minutes > 0) parts.push(minutes + "m");
      if (secs > 0 || parts.length === 0) parts.push(secs + "s");
      return parts.join(" ");
    }}

    // Refresh status every 2 seconds
    function refreshStatus() {{
      fetch('/status')
        .then(response => response.json())
        .then(data => {{
          const timerDisplay = document.getElementById('timer-display');
          const stateText = document.getElementById('status-state');
          const detailsText = document.getElementById('status-details');
          const indicator = document.getElementById('status-indicator');

          if (stateText && data.state) {{
            stateText.textContent = data.state;
          }}
          if (detailsText) {{
            detailsText.textContent = data.details || '';
          }}
          if (indicator && data.state) {{
            const colors = {{
              running: '{p["success"]}',
              starting: '{p["warning"]}',
              stopped: '{p["text_muted"]}',
              paused: '{p["warning"]}',
              waiting: '{p["warning"]}',
              error: '{p["error"]}'
            }};
            indicator.style.background = colors[data.state] || '{p["text"]}';
          }}

          // Update timer display if timer is enabled
          if (data.timer_enabled) {{
            const now = Date.now() / 1000;
            const elapsed = Math.floor(now - data.timer_start_time);
            const remaining = Math.max(0, data.timer_duration_seconds - elapsed);
            const progress = (elapsed / data.timer_duration_seconds * 100).toFixed(1);

            // Color based on remaining time
            let color = '#1DB954'; // green
            if (remaining <= 300) color = '#E74C3C'; // red if < 5min
            else if (remaining <= 600) color = '#F39C12'; // yellow if < 10min

            timerDisplay.innerHTML = `⏱️ Timer: <span style="color: ${{color}};">${{formatTime(remaining)}}</span> | Progress: ${{progress}}% (${{formatTime(elapsed)}} / ${{formatTime(data.timer_duration_seconds)}})`;
            timerDisplay.style.display = 'block';
          }} else {{
            timerDisplay.style.display = 'none';
          }}
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
