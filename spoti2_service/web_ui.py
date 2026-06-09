"""HTML rendering for the Spoti2 web UI."""

from html import escape
from typing import Any, Dict

from spotify_splitter.user_config import DEFAULT_CONFIG


PALETTE = {
    "bg": "#0b0f14",
    "panel": "#161b22",
    "panel_alt": "#11161d",
    "border": "#30363d",
    "accent": "#1DB954",
    "accent_hover": "#1ED760",
    "text": "#e6edf3",
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

    # Live state (badge, track, details) is populated client-side from /status;
    # only config values are server-rendered, and those go through attr()/text().
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Spytify-LE - Linux Audio Recorder</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      background: {p["bg"]};
      color: {p["text"]};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
    }}

    /* Sidebar */
    .sidebar {{
      width: 220px;
      min-height: 100vh;
      background: {p["panel_alt"]};
      border-right: 1px solid {p["border"]};
      padding: 1.5rem 1rem;
      display: flex;
      flex-direction: column;
      position: sticky;
      top: 0;
    }}
    .brand {{ color: {p["accent"]}; font-size: 1.25rem; font-weight: 700; }}
    .brand-sub {{ color: {p["text_muted"]}; font-size: 0.8rem; margin-bottom: 2rem; }}
    .nav-item {{
      display: flex; align-items: center; gap: 0.6rem;
      padding: 0.6rem 0.75rem; margin-bottom: 0.25rem;
      color: {p["text_muted"]}; border-radius: 6px; cursor: pointer;
      border-left: 3px solid transparent; font-weight: 500; user-select: none;
    }}
    .nav-item:hover {{ color: {p["text"]}; background: rgba(255,255,255,0.04); }}
    .nav-item.active {{
      color: {p["accent"]}; background: rgba(29,185,84,0.10);
      border-left-color: {p["accent"]};
    }}
    .sidebar-footer {{
      margin-top: auto; display: flex; align-items: center; gap: 0.5rem;
      color: {p["text_muted"]}; font-size: 0.85rem;
    }}
    .dot {{ width: 9px; height: 9px; border-radius: 50%; background: {p["success"]}; }}

    /* Main */
    .main {{ flex: 1; padding: 1.5rem 2rem; max-width: 1100px; }}
    .view {{ display: none; }}
    .view.active {{ display: block; }}

    .panel {{
      background: {p["panel"]};
      border: 1px solid {p["border"]};
      border-radius: 10px;
      padding: 1.5rem;
      margin-bottom: 1.25rem;
    }}
    .panel h2 {{ font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; }}

    /* Record top row */
    .record-top {{ display: flex; gap: 1.25rem; align-items: stretch; flex-wrap: wrap; }}
    .np-card {{ flex: 1 1 420px; display: flex; gap: 1.25rem; }}
    .controls-card {{ flex: 0 0 250px; display: flex; flex-direction: column; gap: 0.75rem; }}

    .np-art {{
      width: 120px; height: 120px; border-radius: 8px; object-fit: cover;
      background: radial-gradient(circle at 50% 50%, #2a2f37 0%, #14181e 70%);
      border: 1px solid {p["border"]}; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      color: {p["text_muted"]}; font-size: 2rem;
    }}
    .np-info {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
    .np-head {{ display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }}
    .badge {{ display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em; }}
    .badge .dot {{ animation: pulse 2s infinite; }}
    .np-format {{ color: {p["text_muted"]}; font-size: 0.85rem; font-variant-numeric: tabular-nums; }}
    .np-title {{ font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .np-artist {{ color: {p["text_muted"]}; }}
    .np-progress {{ margin-top: auto; padding-top: 1rem; }}
    .np-times {{ display: flex; justify-content: space-between; font-size: 0.75rem; color: {p["text_muted"]}; font-variant-numeric: tabular-nums; margin-bottom: 0.35rem; }}
    .bar {{ height: 4px; background: {p["border"]}; border-radius: 2px; overflow: hidden; }}
    .bar-fill {{ height: 100%; width: 0%; background: {p["accent"]}; transition: width 0.5s linear; }}

    @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.45; }} }}

    /* Buttons */
    button {{
      width: 100%; padding: 0.8rem 1.25rem;
      background: linear-gradient(135deg, {p["accent"]}, {p["accent_hover"]});
      border: none; border-radius: 8px; color: #06210f;
      font-size: 0.95rem; font-weight: 700; cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }}
    button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(29,185,84,0.25); }}
    button.secondary {{
      background: {p["panel_alt"]}; border: 1px solid {p["border"]}; color: {p["text"]};
    }}
    button.secondary:hover {{ background: {p["border"]}; box-shadow: none; }}
    .btn-row {{ display: flex; gap: 0.75rem; }}

    /* Toggle */
    .toggle-row {{ display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; padding-top: 0.5rem; border-top: 1px solid {p["border"]}; }}
    .switch {{ position: relative; display: inline-block; width: 42px; height: 22px; }}
    .switch input {{ opacity: 0; width: 0; height: 0; }}
    .slider {{ position: absolute; cursor: pointer; inset: 0; background: {p["border"]}; border-radius: 22px; transition: 0.2s; }}
    .slider:before {{ content: ""; position: absolute; height: 16px; width: 16px; left: 3px; bottom: 3px; background: #fff; border-radius: 50%; transition: 0.2s; }}
    .switch input:checked + .slider {{ background: {p["accent"]}; }}
    .switch input:checked + .slider:before {{ transform: translateX(20px); }}

    /* Activity log */
    .log-head {{ display: flex; align-items: center; justify-content: space-between; }}
    .log-container {{
      background: {p["bg"]}; border: 1px solid {p["border"]}; border-radius: 8px;
      padding: 0.75rem; max-height: 420px; overflow-y: auto;
      font-family: ui-monospace, "Cascadia Code", "Courier New", monospace; font-size: 0.85rem;
    }}
    .log-line, .log-success, .log-error, .log-warning, .log-info, .log-track, .log-waiting {{
      padding: 0.4rem 0.6rem; margin-bottom: 0.4rem; border-left: 3px solid transparent;
      border-radius: 4px; line-height: 1.5;
    }}
    .log-success {{ background: rgba(29,185,84,0.08); border-left-color: {p["success"]}; color: {p["success"]}; }}
    .log-error {{ background: rgba(248,81,73,0.10); border-left-color: {p["error"]}; color: {p["error"]}; }}
    .log-warning {{ background: rgba(210,153,34,0.10); border-left-color: {p["warning"]}; color: {p["warning"]}; }}
    .log-info {{ background: rgba(29,185,84,0.04); border-left-color: {p["accent"]}; color: {p["text"]}; }}
    .log-track {{ background: rgba(29,185,84,0.06); border-left-color: {p["accent"]}; color: {p["accent"]}; font-weight: 500; }}
    .log-line {{ background: rgba(255,255,255,0.02); border-left-color: {p["border"]}; color: {p["text_muted"]}; }}
    .log-waiting {{ color: {p["text_muted"]}; text-align: center; padding: 2rem; font-style: italic; }}

    /* Recorded tracks table */
    table.history {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.history th, table.history td {{ text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid {p["border"]}; }}
    table.history th {{ color: {p["text_muted"]}; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    table.history td.hist-icon {{ width: 1.5rem; text-align: center; }}
    table.history th.hist-year, table.history td.hist-year {{ width: 4rem; font-variant-numeric: tabular-nums; }}
    table.history td .hist-artist {{ color: {p["text_muted"]}; }}
    table.history td.hist-empty {{ text-align: center; color: {p["text_muted"]}; font-style: italic; padding: 1.5rem; }}
    tr.hist-failed td {{ color: {p["error"]}; }}
    tr.hist-skipped td {{ color: {p["text_muted"]}; }}

    /* Forms (Settings / Advanced) */
    .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; }}
    label {{ display: block; margin-bottom: 1rem; }}
    label span {{ display: block; margin-bottom: 0.4rem; font-weight: 500; }}
    input[type="text"], select {{
      width: 100%; padding: 0.7rem; background: {p["bg"]}; border: 1px solid {p["border"]};
      border-radius: 6px; color: {p["text"]}; font-size: 0.95rem;
    }}
    input[type="text"]:focus, select:focus {{ outline: none; border-color: {p["accent"]}; }}
    .checkbox-group {{ display: flex; flex-wrap: wrap; gap: 1.25rem; margin: 1rem 0; }}
    .checkbox-group label {{ display: flex; align-items: center; gap: 0.5rem; margin: 0; }}
    input[type="checkbox"] {{ width: 17px; height: 17px; cursor: pointer; accent-color: {p["accent"]}; }}
    .help-text {{ color: {p["text_muted"]}; font-size: 0.85rem; margin-top: 0.4rem; }}
    .save-btn {{ width: auto; padding: 0.6rem 1.5rem; margin-top: 0.5rem; }}
  </style>
</head>
<body>
  <nav class="sidebar">
    <div class="brand">Spytify-LE</div>
    <div class="brand-sub">Linux Audio Recorder</div>
    <div class="nav-item active" data-view="record" onclick="switchView('record', this)">🎙️ Record</div>
    <div class="nav-item" data-view="settings" onclick="switchView('settings', this)">⚙️ Settings</div>
    <div class="nav-item" data-view="advanced" onclick="switchView('advanced', this)">🛠️ Advanced</div>
    <div class="sidebar-footer"><span class="dot"></span><span id="system-state">System Ready</span></div>
  </nav>

  <main class="main">
    <!-- Record -->
    <section id="view-record" class="view active">
      <div class="record-top">
        <div class="panel np-card">
          <div id="np-art" class="np-art">◉</div>
          <div class="np-info">
            <div class="np-head">
              <span class="badge"><span id="np-dot" class="dot"></span><span id="np-badge">STOPPED</span></span>
              <span id="np-format" class="np-format"></span>
            </div>
            <div id="np-title" class="np-title">Waiting for playback…</div>
            <div id="np-artist" class="np-artist"></div>
            <div class="np-progress">
              <div class="np-times"><span id="np-elapsed">00:00</span><span id="np-duration">00:00</span></div>
              <div class="bar"><div id="np-bar" class="bar-fill"></div></div>
            </div>
          </div>
        </div>

        <div class="panel controls-card">
          <form method="post" action="/start"><button type="submit">▶ Start Recording</button></form>
          <div class="btn-row">
            <form method="post" action="/pause" style="flex:1;"><button type="submit" class="secondary">❚❚ Pause</button></form>
            <form method="post" action="/stop" style="flex:1;"><button type="submit" class="secondary">■ Stop</button></form>
          </div>
          <form method="post" action="/update" class="toggle-row">
            <span>Overwrite existing</span>
            <label class="switch">
              <input type="hidden" name="allow_overwrite" value="0">
              <input type="checkbox" name="allow_overwrite" value="on" {checked(config.get("allow_overwrite", False))} onchange="this.form.submit()">
              <span class="slider"></span>
            </label>
          </form>
        </div>
      </div>

      <div class="panel">
        <div class="log-head"><h2>Recording Activity</h2>
          <form method="post" action="/toggle-verbose" style="margin:0;">
            <label style="display:flex; align-items:center; gap:0.4rem; margin:0; font-size:0.8rem; color:{p["text_muted"]};">
              <input type="checkbox" name="verbose" {checked(verbose_logging)} onchange="this.form.submit()"> verbose
            </label>
          </form>
        </div>
        <div class="log-container" id="log-display"><div class="log-waiting">Loading logs…</div></div>
        <div id="timer-display" class="help-text" style="display:none; margin-top:0.75rem; font-weight:600;"></div>
      </div>

      <div class="panel">
        <h2>Recorded Tracks</h2>
        <table class="history">
          <thead><tr><th class="hist-icon"></th><th>Track</th><th class="hist-year">Year</th><th>Genre</th></tr></thead>
          <tbody id="history-rows"><tr><td colspan="4" class="hist-empty">No tracks recorded yet…</td></tr></tbody>
        </table>
      </div>
    </section>

    <!-- Settings -->
    <section id="view-settings" class="view">
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

        <div class="checkbox-group">
          <label>
            <input type="hidden" name="allow_overwrite" value="0">
            <input type="checkbox" name="allow_overwrite" value="on" {checked(config.get("allow_overwrite", False))}>
            <span>Allow Overwriting Existing Files</span>
          </label>
        </div>
        <div class="help-text">Re-record tracks even if they already exist (useful for fixing incomplete recordings)</div>

        <h2 style="margin-top:2rem;">Metadata</h2>
        <div class="form-grid">
          <label>
            <span>LastFM API Key</span>
            <input type="text" name="lastfm_api_key" value="{attr(config.get("lastfm_api_key", "") or "")}" placeholder="Enter your LastFM API key" />
            <div class="help-text">Required for year and genre tags. <a href="https://www.last.fm/api/account/create" target="_blank" style="color:{p["accent"]};">Get one here</a></div>
          </label>
        </div>

        <h2 style="margin-top:2rem;">Playlist</h2>
        <div class="form-grid">
          <label>
            <span>Playlist File (optional)</span>
            <input type="text" name="playlist" value="{attr(config.get("playlist", "") or "")}" placeholder="/path/to/playlist.m3u" />
            <div class="help-text">Generate an M3U playlist file</div>
          </label>
          <label>
            <span>Bundle Album Artwork URL (optional)</span>
            <input type="text" name="bundle_album_art_uri" value="{attr(config.get("bundle_album_art_uri", "") or "")}" placeholder="https://example.com/album-cover.jpg" />
            <div class="help-text">Custom artwork for bundle playlists</div>
          </label>
          <label>
            <span>M3U Playlist Base Path (optional)</span>
            <input type="text" name="playlist_base_path" value="{attr(config.get("playlist_base_path", "") or "")}" placeholder="/mnt/storage/music" />
            <div class="help-text">Maps local recording paths to remote server paths in the playlist</div>
          </label>
        </div>
        <div class="checkbox-group">
          <label>
            <input type="hidden" name="bundle_playlist" value="0">
            <input type="checkbox" name="bundle_playlist" value="on" {checked(config.get("bundle_playlist", False))}>
            <span>Bundle as Compilation Album</span>
          </label>
        </div>

        <button type="submit" class="save-btn">Save Settings</button>
      </form>
    </section>

    <!-- Advanced -->
    <section id="view-advanced" class="view">
      <form class="panel" method="post" action="/update">
        <h2>Performance</h2>
        <label>
          <span>Configuration Profile</span>
          <select name="profile">
            {"".join(_select_options(config.get("profile", "auto"), ["auto","desktop","headless","high_performance"]))}
          </select>
          <div class="help-text">Optimize for your system</div>
        </label>
        <div class="checkbox-group" style="margin-top:1rem;">
          <label>
            <input type="hidden" name="enable_adaptive" value="0">
            <input type="checkbox" name="enable_adaptive" value="on" {checked(config.get("enable_adaptive", True))}>
            <span>Adaptive Buffers</span>
          </label>
          <label>
            <input type="hidden" name="enable_monitoring" value="0">
            <input type="checkbox" name="enable_monitoring" value="on" {checked(config.get("enable_monitoring", False))}>
            <span>Buffer Monitoring (debug)</span>
          </label>
          <label>
            <input type="hidden" name="debug_mode" value="0">
            <input type="checkbox" name="debug_mode" value="on" {checked(config.get("debug_mode", False))}>
            <span>Debug Mode</span>
          </label>
        </div>

        <h2 style="margin-top:2rem;">Player</h2>
        <label>
          <span>MPRIS Player Name</span>
          <input type="text" name="player" value="{attr(config.get("player", DEFAULT_CONFIG["player"]))}" />
          <div class="help-text">Usually "spotify" for the Spotify desktop client</div>
        </label>
        <button type="submit" class="save-btn">Save Advanced Settings</button>
      </form>
    </section>
  </main>

  <script>
    const COLORS = {{
      running: '{p["success"]}', starting: '{p["warning"]}', stopped: '{p["text_muted"]}',
      paused: '{p["warning"]}', waiting: '{p["warning"]}', error: '{p["error"]}'
    }};
    const BADGE = {{
      running: 'RECORDING ACTIVE', starting: 'STARTING', stopped: 'STOPPED',
      paused: 'PAUSED', waiting: 'WAITING', error: 'ERROR'
    }};

    function switchView(name, el) {{
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      document.getElementById('view-' + name).classList.add('active');
      el.classList.add('active');
    }}

    function refreshLogs() {{
      fetch('/logs').then(r => r.json()).then(data => {{
        const el = document.getElementById('log-display');
        if (data.logs) {{ el.innerHTML = data.logs; el.scrollTop = 0; }}
      }}).catch(e => console.error('logs', e));
    }}

    function fmtTime(sec) {{
      sec = Math.max(0, Math.floor(sec));
      const m = Math.floor(sec / 60), s = sec % 60;
      return String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
    }}

    // Client-side progress interpolation (position is only sampled at track change).
    let npKey = null, npStart = 0, npDuration = 0;

    function paintProgress() {{
      if (npDuration <= 0) {{
        document.getElementById('np-bar').style.width = '0%';
        document.getElementById('np-elapsed').textContent = '00:00';
        document.getElementById('np-duration').textContent = '00:00';
        return;
      }}
      let elapsed = Math.min(Math.max(Date.now()/1000 - npStart, 0), npDuration);
      document.getElementById('np-bar').style.width = (elapsed / npDuration * 100) + '%';
      document.getElementById('np-elapsed').textContent = fmtTime(elapsed);
      document.getElementById('np-duration').textContent = fmtTime(npDuration);
    }}

    function refreshStatus() {{
      fetch('/status').then(r => r.json()).then(data => {{
        const st = data.state || 'unknown';
        document.getElementById('np-badge').textContent = BADGE[st] || st.toUpperCase();
        document.getElementById('np-dot').style.background = COLORS[st] || '{p["text"]}';
        document.getElementById('system-state').textContent =
          (st === 'running') ? 'Recording' : (data.details || 'System Ready');

        const track = data.track || {{}};
        const title = track.title || (st === 'running' ? 'Recording…' : 'Waiting for playback…');
        document.getElementById('np-title').textContent = title;
        document.getElementById('np-artist').textContent = track.artist || '';

        const art = document.getElementById('np-art');
        if (track.art_uri) {{
          art.style.backgroundImage = 'url("' + track.art_uri + '")';
          art.style.backgroundSize = 'cover';
          art.textContent = '';
        }} else {{
          art.style.backgroundImage = '';
          art.textContent = '◉';
        }}

        const fmt = (data.output_format || '').toUpperCase();
        const rate = data.samplerate ? (data.samplerate / 1000).toFixed(1) + ' kHz' : '';
        document.getElementById('np-format').textContent = [fmt, rate].filter(Boolean).join('  |  ');

        // Reset interpolation when the track changes; position is microseconds.
        const key = (track.artist || '') + ' - ' + (track.title || '');
        npDuration = (track.duration_ms || 0) / 1000;
        if (key !== npKey) {{
          npKey = key;
          npStart = Date.now()/1000 - (track.position || 0) / 1e6;
        }}
        paintProgress();

        // Timer (max-duration) readout
        const timer = document.getElementById('timer-display');
        if (data.timer_enabled) {{
          const remaining = Math.max(0, data.timer_remaining_seconds || 0);
          let color = '{p["success"]}';
          if (remaining <= 300) color = '{p["error"]}'; else if (remaining <= 600) color = '{p["warning"]}';
          timer.innerHTML = '⏱️ Timer: <span style="color:' + color + ';">' + fmtTime(remaining) + '</span> remaining';
          timer.style.display = 'block';
        }} else {{
          timer.style.display = 'none';
        }}
      }}).catch(e => console.error('status', e));
    }}

    const HIST_ICON = {{
      saved: '✅', skipped_incomplete: '⏭️', skipped_exists: '⏭️', failed: '❌'
    }};

    function refreshHistory() {{
      fetch('/history').then(r => r.json()).then(data => {{
        const tbody = document.getElementById('history-rows');
        const records = data.records || [];
        if (!records.length) {{
          tbody.innerHTML = '<tr><td colspan="4" class="hist-empty">No tracks recorded yet…</td></tr>';
          return;
        }}
        tbody.replaceChildren();
        for (const rec of records) {{
          const tr = document.createElement('tr');
          const outcome = rec.outcome || '';
          if (outcome === 'failed') tr.className = 'hist-failed';
          else if (outcome.startsWith('skipped')) tr.className = 'hist-skipped';

          const icon = document.createElement('td');
          icon.className = 'hist-icon';
          icon.textContent = HIST_ICON[outcome] || '•';
          icon.title = (outcome + (rec.reason ? ': ' + rec.reason : ''));
          tr.appendChild(icon);

          const track = document.createElement('td');
          const title = document.createElement('span');
          title.textContent = rec.title || '(unknown)';
          const artist = document.createElement('span');
          artist.className = 'hist-artist';
          artist.textContent = rec.artist ? '  —  ' + rec.artist : '';
          track.appendChild(title); track.appendChild(artist);
          tr.appendChild(track);

          const year = document.createElement('td');
          year.className = 'hist-year';
          year.textContent = rec.year || '';
          tr.appendChild(year);

          const genre = document.createElement('td');
          genre.textContent = rec.genre || '';
          tr.appendChild(genre);

          tbody.appendChild(tr);
        }}
      }}).catch(e => console.error('history', e));
    }}

    refreshLogs(); refreshStatus(); refreshHistory();
    setInterval(refreshHistory, 5000);
    setInterval(refreshLogs, 3000);
    setInterval(refreshStatus, 2000);
    setInterval(paintProgress, 1000);
  </script>
</body>
</html>
""".strip()
