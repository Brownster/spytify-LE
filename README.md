# Spoti2 - Linux Spotify Desktop Recorder

Record Spotify desktop playback on Linux and save each track as an individual audio file with rich metadata from LastFM.

<img width="1863" height="217" alt="image" src="https://github.com/user-attachments/assets/d9344323-0293-4681-9761-9abfb30f2c28" />

## Features

- 🎵 **Automatic Track Splitting** - Monitors Spotify desktop via MPRIS and splits tracks automatically
- 🎨 **Rich Metadata** - Fetches year and genre from LastFM, plus album art from Spotify
- 🌐 **Modern Web UI** - Beautiful 3-tab interface for easy control and monitoring with minimal, clean logs
- ⚡ **High Quality** - Records lossless via PulseAudio/PipeWire monitor sources
- 📝 **Playlist Support** - Generate M3U playlists with optional bundling
- 🚫 **Ad Filtering** - Automatically skips advertisements
- ⏯️ **Smart Controls** - Start, stop, pause, and resume recording from web UI
- 🔄 **Auto-Recovery** - Robust error handling and automatic restarts
- 🔁 **Overwrite Control** - Optional re-recording of existing tracks

## Prerequisites

- Linux with PulseAudio or PipeWire
- Spotify desktop client (tested with [Flatpak version](https://flathub.org/apps/com.spotify.Client))
- Python 3.10– <4.0
- System packages: `python3-pyaudio` and `ffmpeg`
- **LastFM API key** (free) - [Get one here](https://www.last.fm/api/account/create)

**For source installation:**
- [Poetry](https://python-poetry.org/) installed

## Installation

### Option 1: Install from Release (Recommended)

Download the latest `.whl` file from [GitHub Releases](https://github.com/Brownster/spytify-LE/releases):

```bash
# Install system dependencies
sudo apt install python3-pip python3-pyaudio ffmpeg  # Ubuntu/Debian
# OR
sudo dnf install python3-pip python3-pyaudio ffmpeg  # Fedora

# Install spotify-splitter
pip install --user spotify_splitter-0.1.0-py3-none-any.whl

# Add to PATH if needed
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Install from Source

```bash
git clone https://github.com/Brownster/spytify-LE.git
cd spytify-LE
sudo apt install python3-pyaudio ffmpeg
poetry config virtualenvs.options.system-site-packages true
poetry install
```

## Quick Start

### Web UI (Recommended)

Start the web service for the easiest experience:

```bash
# Start the web interface (runs on http://localhost:8730)
python -m spoti2_service

# Or customize host/port
python -m spoti2_service --host 0.0.0.0 --port 8080
```

Then open http://localhost:8730 in your browser. You'll see:

- **Record Tab** - Start/Stop/Pause recording, live status, and recording log
- **Settings Tab** - Configure output directory, format, and LastFM API key
- **Advanced Tab** - Performance settings and player configuration

### CLI Usage

```bash
# Interactive setup wizard (recommended for first-time use)
spotify-splitter configure

# Basic recording
spotify-splitter record

# Custom output directory and format
spotify-splitter --output ~/Music/Spotify --format flac record

# Create playlist of recorded tracks
spotify-splitter record --playlist mysession.m3u

# Bundle playlist tracks as single album
spotify-splitter record --playlist mysession.m3u --bundle-playlist
```

By default, tracks are saved to `~/Music/Spotify Splitter/<Artist>/<Album>/<Track>.mp3` with full ID3 tags including:
- Artist, Title, Album, Track Number
- Album Art (embedded JPEG)
- **Year** (from LastFM)
- **Genre** (from LastFM)

## Configuration

### LastFM API Key Setup

1. **Get your API key** at https://www.last.fm/api/account/create
2. **Add it to the config**:
   - Via Web UI: Settings tab → LastFM API Key field
   - Via CLI: Edit `~/.config/spotify_splitter/config.json`
   - Or set in code: `spotify_splitter/lastfm_api.py` line 18

### Configuration Profiles

Spoti2 includes optimized profiles for different use cases:

- **auto** - Automatically detects optimal settings
- **desktop** - Balanced for desktop usage (default)
- **headless** - Optimized for servers/headless systems
- **high_performance** - Low-latency for powerful systems

```bash
spotify-splitter record --profile headless
```

### Advanced Options

```bash
# Buffer tuning for lower-end systems
spotify-splitter record --queue-size 50 --latency 0.1

# Disable performance monitoring
spotify-splitter record --no-adaptive --no-monitoring

# Debug mode with performance dashboard
spotify-splitter record --debug-mode
```

## Web UI Features

The modern web interface provides:

### Record Tab
- **Live Status** - See recording state with clear indicators showing current track
- **Control Buttons** - Start, Stop, Pause, Resume recording
- **Recording Log** - Clean, minimal logs showing only essential events:
  - Tracks saved with file paths
  - Critical errors
  - Duplicate file detections
- **Verbose Toggle** - Optional checkbox to show detailed logs for troubleshooting:
  - Track changes
  - MPRIS events
  - Warnings and buffer status

### Settings Tab
- **Output directory configuration** - Where tracks are saved
- **Audio format selection** - MP3, FLAC, WAV, OGG
- **Overwrite existing files** - Toggle to re-record tracks that already exist (useful for fixing incomplete recordings)
- **LastFM API key management** - Required for year and genre metadata
- **Playlist generation options** - Create M3U playlists with optional bundling

### Advanced Tab
- **Performance profile selection** - Auto, desktop, headless, high_performance
- **Buffer management settings** - Adaptive buffers, monitoring, metrics
- **Player name configuration** - MPRIS player selection
- **Debug options** - Advanced diagnostics

## Service Mode & Systemd Integration

Run Spoti2 as a background service with automatic restarts:

```bash
# Start service manually
python -m spoti2_service
```

### Example systemd Unit

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

## Library Management Integration

### Lidarr Integration

Point [Lidarr](https://lidarr.audio/)'s "Manual Import" folder to your output directory for automatic organization:

1. Configure Lidarr to monitor `~/Music/Spotify Splitter`
2. Lidarr will automatically:
   - Match tracks to its database
   - Fetch additional metadata
   - Move files to organized library
   - Handle duplicates

### Music Players

Recorded tracks work perfectly with:
- **Strawberry** - Advanced music player with excellent metadata support
- **Rhythmbox** - GNOME's default music player
- **Clementine** - Feature-rich player with smart playlists
- **VLC** - Universal media player

<img width="400" height="764" alt="image" src="https://github.com/user-attachments/assets/e2cebd73-57da-4d80-a896-90ffa5c6c804" />

## Troubleshooting

### No Audio Device Found

If you see `ValueError: No input device matching`:

```bash
# List available monitors
pactl list sources | grep -E "Name:|Description:"

# Find your Spotify monitor (usually contains "spotify" or "Monitor")
# Update player name if needed
spotify-splitter record --player spotify
```

### Audio Quality Issues

If tracks sound distorted or have wrong speed:

1. **Check sample rate**: Ensure monitor source matches Spotify output
2. **Verify audio backend**: PulseAudio or PipeWire should be running
3. **Try different latency**: `spotify-splitter record --latency 0.05`

### Buffer Overflow Warnings

If you see "input overflow" warnings:

```bash
# Increase buffer size
spotify-splitter record --queue-size 50 --latency 0.1

# Or use headless profile
spotify-splitter record --profile headless
```

### LastFM Metadata Not Working

1. Verify API key is set correctly
2. Check network connection
3. View logs for API errors: `~/.cache/spotify_splitter/service/recorder.log`

## Architecture Overview

### Core Technology Stack

- **MPRIS D-Bus** - Real-time track metadata from Spotify desktop
- **PulseAudio/PipeWire** - Lossless digital audio capture via monitor sources
- **sounddevice + PortAudio** - Cross-platform audio handling with automatic sample rate detection
- **LastFM API** - Rich metadata enrichment (year, genre)
- **mutagen** - ID3v2.4 tag writing with embedded album art
- **Rich + Typer** - Beautiful CLI interface
- **HTTP Server** - Modern web UI with live updates

### Audio Processing Pipeline

```
Spotify Desktop Audio Output
           ↓
PulseAudio/PipeWire Monitor
           ↓
sounddevice (PortAudio)
           ↓
Adaptive Buffer Management
           ↓
Track Boundary Detection (MPRIS)
           ↓
Format Conversion (float32 → int16)
           ↓
Audio Segmentation per Track
           ↓
ID3 Tagging (Mutagen)
  ├─ Basic: Artist, Title, Album, Track #
  ├─ Album Art (from Spotify)
  └─ LastFM: Year, Genre
           ↓
Export (MP3/FLAC/WAV/OGG)
```

### Key Features

**Reliability**
- Incomplete track detection
- Advertisement filtering
- Duplicate detection
- Automatic error recovery
- Graceful degradation

**Performance**
- Adaptive buffer sizing
- Non-blocking threaded processing
- Memory-efficient streaming
- Dynamic latency adjustment
- Real-time performance monitoring

**Production Ready**
- Comprehensive test suite
- Type-hinted codebase
- Modular architecture
- Extensive logging
- Web-based monitoring

## Contributing

Contributions welcome! This project focuses on:
- Linux Spotify desktop client support
- High-quality audio capture
- Rich metadata via LastFM
- Clean, maintainable code

## License

MIT License - See LICENSE file for details

## Credits

Inspired by the excellent [spy-spotify](https://github.com/jwallet/spy-spotify) Windows application.
