# Installation Guide - Spoti2

Quick installation guide for Spoti2, the Linux Spotify Desktop recorder.

## Prerequisites

- Linux with PulseAudio or PipeWire
- Spotify desktop client (Flatpak or native)
- Python 3.10 or higher
- System packages: `python3-pyaudio` and `ffmpeg`

## Quick Install

### Method 1: Using pip (Recommended)

```bash
# 1. Install system dependencies
# Ubuntu/Debian
sudo apt install python3-pip python3-pyaudio ffmpeg

# Fedora
sudo dnf install python3-pip python3-pyaudio ffmpeg

# 2. Install spotify-splitter
pip install --user spotify_splitter-0.1.0-py3-none-any.whl

# 3. Add to PATH (if needed)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: From Source

```bash
# 1. Install system dependencies
sudo apt install python3-pyaudio ffmpeg  # Ubuntu/Debian
# OR
sudo dnf install python3-pyaudio ffmpeg  # Fedora

# 2. Clone and install
git clone https://github.com/Brownster/spytify-LE.git
cd spytify-LE
poetry config virtualenvs.options.system-site-packages true
poetry install

# 3. Verify installation
poetry run spotify-splitter --help
```

## First-Time Setup

### 1. Get LastFM API Key (Free)

1. Go to https://www.last.fm/api/account/create
2. Register an application (any name/description is fine)
3. Copy your API Key

### 2. Configure Spoti2

**Option A: Web UI (Easiest)**
```bash
# Start web interface
python -m spoti2_service

# Open http://localhost:8730
# Go to Settings tab and enter your LastFM API key
```

**Option B: Interactive CLI**
```bash
# Run configuration wizard
spotify-splitter configure

# Follow the prompts to set:
# - Output directory
# - Audio format (MP3, FLAC, WAV, OGG)
# - Player name (usually "spotify")
```

**Option C: Manual Config**
```bash
# Edit config file directly
nano ~/.config/spotify_splitter/config.json

# Add your LastFM API key:
{
  "lastfm_api_key": "YOUR_API_KEY_HERE",
  "output": "/home/you/Music/Spotify Splitter",
  "format": "mp3"
}
```

## Usage

### Web UI Mode (Recommended)

```bash
# Start the web service
python -m spoti2_service

# Open http://localhost:8730 in browser
# Use Start/Stop/Pause buttons to control recording
```

### CLI Mode

```bash
# Basic recording
spotify-splitter record

# Custom output and format
spotify-splitter --output ~/Music/Spotify --format flac record

# Create playlist
spotify-splitter record --playlist mysession.m3u

# Bundle playlist as single album
spotify-splitter record --playlist mysession.m3u --bundle-playlist
```

### Running as Service (Optional)

Create `~/.config/systemd/user/spoti2.service`:

```ini
[Unit]
Description=Spoti2 Recording Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/spoti2
ExecStart=/usr/bin/python -m spoti2_service
Restart=on-failure
RestartSec=15
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

Enable and start:
```bash
systemctl --user enable spoti2.service
systemctl --user start spoti2.service
systemctl --user status spoti2.service
```

## Troubleshooting

### No Input Device Found

```bash
# List available audio sources
pactl list sources | grep -E "Name:|Description:"

# Look for Spotify monitor (contains "spotify" or "Monitor")
# If needed, update player name:
spotify-splitter record --player spotify
```

### Command Not Found

```bash
# Add ~/.local/bin to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Spotify Not Detected

1. Make sure Spotify desktop is running
2. Start playing a track
3. Check MPRIS is working: `qdbus org.mpris.MediaPlayer2.spotify`
4. If no output, Spotify MPRIS might be disabled

### Buffer Overflow Warnings

```bash
# Increase buffer size
spotify-splitter record --queue-size 50 --latency 0.1

# Or use headless profile for stability
spotify-splitter record --profile headless
```

### LastFM Metadata Not Fetched

1. Verify API key is correct in settings
2. Check internet connection
3. View logs: `~/.cache/spotify_splitter/service/recorder.log`
4. Try testing the API key at https://www.last.fm/api/

## Verifying Installation

```bash
# Check CLI works
spotify-splitter --help

# Check web service works
python -m spoti2_service --help

# Test recording (5 seconds)
spotify-splitter record
# Play a song in Spotify, wait 5 seconds, press Ctrl+C
# Check output in ~/Music/Spotify Splitter/
```

## Uninstallation

```bash
# Remove pip installation
pip uninstall spotify-splitter

# Remove config
rm -rf ~/.config/spotify_splitter

# Remove cache/logs
rm -rf ~/.cache/spotify_splitter

# Remove systemd service (if installed)
systemctl --user stop spoti2.service
systemctl --user disable spoti2.service
rm ~/.config/systemd/user/spoti2.service
```

## Next Steps

- Read the [README](README.md) for detailed features and configuration
- Set up [Lidarr](https://lidarr.audio/) for automatic library organization
- Integrate with music players like Strawberry or Rhythmbox
- Report issues at https://github.com/Brownster/spytify-LE/issues
