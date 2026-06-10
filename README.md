# Spytify-LE - Linux Spotify Desktop Recorder

Record Spotify desktop playback on Linux and save each track as an individual audio file with rich metadata from LastFM.

<img width="1702" height="847" alt="image" src="https://github.com/user-attachments/assets/04a508fa-4e8f-4c8c-bab1-ce06e8369a78" />


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

### First Run (Recommended)

Open Spotify, start playing a track, then run:

```bash
# Check system dependencies, Spotify detection, ffmpeg, and output settings
spotify-splitter doctor

# Start the local web UI at http://127.0.0.1:8730
spotify-splitter web
```

`spotify-splitter web` binds to `127.0.0.1` by default and opens your browser.
Use `--no-open` if you do not want it to launch a browser.

The web UI shows readiness checks for common setup problems:

- Spotify not detected
- Spotify detected but not playing
- `ffmpeg` missing
- PulseAudio/PipeWire not reachable
- output folder not writable

### Web UI

The web interface includes:

- **Record Tab** - Start/Stop/Pause recording, live status, readiness checks, and recorded-track history
- **Settings Tab** - Configure output directory, format, and LastFM API key
- **Advanced Tab** - Hidden advanced buffer/debug controls for troubleshooting

### CLI Usage

```bash
# Interactive setup wizard (recommended for first-time use)
spotify-splitter configure

# Basic recording
spotify-splitter record

# Record with an automatic timer
spotify-splitter record --duration "4h29m"

# Custom output directory and format
spotify-splitter --output ~/Music/Spotify --format flac record

# Create playlist of recorded tracks
spotify-splitter record --playlist mysession.m3u

# Bundle playlist tracks as single album
spotify-splitter record --playlist mysession.m3u --bundle-playlist

# Record with automatic timer (stops after specified duration)
spotify-splitter record --max-duration "4h29m"
spotify-splitter record --max-duration "90m" --playlist playlist.m3u
```

#### Recording with Timer

Automatically stop recording after a specified duration. This is perfect for recording Spotify playlists without having to manually monitor when they end:

```bash
# Record for 4 hours and 29 minutes
spotify-splitter record --max-duration "4h29m"

# Record for 90 minutes with playlist creation
spotify-splitter record --max-duration "90m" --playlist my-favorites.m3u

# Other supported formats
spotify-splitter record --max-duration "2h30m"  # Hours and minutes
spotify-splitter record --max-duration "2h"     # Hours only
spotify-splitter record --max-duration "5400s"  # Seconds
```

**How to use:**
1. Start recording with a random song playing
2. Start your Spotify playlist (first track will be partially recorded and can be skipped)
3. Check the playlist duration in Spotify (e.g., "4 hours 28 minutes")
4. Set the timer to that duration plus a minute: `--max-duration "4h29m"`
5. The app will automatically stop recording when the timer expires

The timer displays in the UI with:
- Remaining time (color-coded: green > 10min, yellow > 5min, red < 5min)
- Progress percentage
- Elapsed time vs total time

The recording stops gracefully, ensuring all buffered audio is processed and the current track is saved completely.

By default, tracks are saved to `~/Music/Spotify Splitter/<Artist>/<Album>/<Track>.mp3` with full ID3 tags including:
- Artist, Title, Album, Track Number
- Album Art (embedded JPEG)
- **Year** (from LastFM)
- **Genre** (from LastFM)

## Configuration

### LastFM API Key Setup (Required)

**LastFM API key is required** for year and genre metadata tagging.

1. **Get your free API key** at https://www.last.fm/api/account/create
2. **Add it to the config**:
   - **Via Web UI (Recommended)**: Settings tab → LastFM API Key field → Save Settings
   - **Via Config File**: Edit `~/.config/spotify_splitter/config.json` and add:
     ```json
     {
       "lastfm_api_key": "YOUR_API_KEY_HERE"
     }
     ```

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
- **Readiness Checks** - Run `doctor` checks from the browser before recording
- **Recorded Tracks** - Review saved/skipped/failed tracks and correct year/genre tags inline

### Settings Tab
- **Output directory configuration** - Where tracks are saved
- **Audio format selection** - MP3, FLAC, WAV, OGG
- **Overwrite existing files** - Toggle to re-record tracks that already exist (useful for fixing incomplete recordings)
- **LastFM API key management** - Required for year and genre metadata
- **Playlist generation options** - Create M3U playlists with optional bundling

### Advanced Tab
- **Performance profile selection** - Auto, desktop, headless, high_performance
- **Buffer management settings** - Adaptive buffers and debug-only monitoring
- **Player name configuration** - MPRIS player selection
- **Debug options** - Hidden behind an expandable advanced section

## Service Mode & Systemd Integration

Run Spoti2 as a background service with automatic restarts:

```bash
# Start service manually
spotify-splitter web --no-open
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
ExecStart=/usr/bin/spotify-splitter web --no-open
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

### "Spotify sink not found" (Spotify Flatpak / PipeWire)

Recent Spotify Flatpak builds (≈1.2.89+) connect to audio over PipeWire's
**native protocol**. PipeWire's PulseAudio-compatibility layer then strips the
`application.name`/`spotify` identifiers from the `pactl` sink-input view, so
older versions of this tool failed with `Spotify sink not found – is music playing?`
even while Spotify was clearly playing.

Auto-detection now handles this by cross-referencing `pw-dump` (the native
PipeWire view, which still carries the Spotify identity), so on current versions
it should just work. If detection still can't find the stream, point it at the
capture source explicitly:

```bash
# Inspect what PipeWire/PulseAudio expose
pw-dump | grep -A2 '"application.name": "spotify"'   # the native node
pactl list sink-inputs | grep node.name              # its pulse-compat node.name
pactl list sources short                             # monitor sources

# Record from a specific source/device (PortAudio device name or PulseAudio source)
spotify-splitter record --monitor audio-src
spotify-splitter record --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
```

> **Note:** the capturable PortAudio device name may differ from the PulseAudio
> source name. On native-PipeWire setups the Spotify stream node (e.g. `audio-src`)
> is often what PortAudio can open, while the `.monitor` name may not resolve —
> if one fails to open, try the other.

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
