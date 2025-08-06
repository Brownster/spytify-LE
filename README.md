# Spotify Splitter

This project records Spotify playback on Linux and saves each track as an individual audio file with metadata.

<img width="1863" height="217" alt="image" src="https://github.com/user-attachments/assets/d9344323-0293-4681-9761-9abfb30f2c28" />

## Prerequisites

- Linux with PulseAudio or PipeWire
- A running Spotify client
- Python 3.10– <4.0
- `python3-pyaudio` and `ffmpeg` (system packages)

**For source installation only:**
- [Poetry](https://python-poetry.org/) installed

ffmpeg is required for converting audio during export and when running tests.

Tested with flatpack official app

https://flathub.org/apps/com.spotify.Client

## Installation

### Option 1: Install from Release (Recommended)

Download the latest `.whl` file from [GitHub Releases](https://github.com/Brownster/spoti2/releases) and install:

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

After installation, you can run `spotify-splitter` from anywhere without `poetry run`.

### Option 2: Install from Source

```bash
git clone https://github.com/Brownster/spoti2.git
cd spoti2
sudo apt install python3-pyaudio ffmpeg
poetry config virtualenvs.options.system-site-packages true
poetry install
```

### With spotifyd (Headless)

For headless operation, you can use [spotifyd](https://github.com/Spotifyd/spotifyd):

1. **Install spotifyd** (check your distribution's package manager)
2. **Configure spotifyd** at `~/.config/spotifyd/spotifyd.conf`:
   ```
   [global]
   username = "your_username"
   password = "your_password"
   device_name = "YourDevice"
   backend = "pulseaudio"
   volume_normalisation = true
   bitrate = 320
   use_mpris = true
   ```
3. **Start spotifyd**: `systemctl --user start spotifyd`
4. **Run spotify-splitter**: 
   - With wheel: `spotify-splitter --output ~/Music record --spotifyd-mode`
   - With source: `poetry run spotify-splitter --output ~/Music record --spotifyd-mode`

### Enabling MPRIS on spotifyd

Spotify Splitter receives track metadata via the [MPRIS](https://specifications.freedesktop.org/mpris-spec/latest/) D-Bus interface. When running `spotifyd` on headless systems, no session bus may be available. You can either launch `spotifyd` with its own bus using `dbus-launch` or configure it to use the system bus.

**Option 1 – dbus-launch**

Create a wrapper script and start it with `dbus-launch`:

```bash
#!/bin/bash
echo "$DBUS_SESSION_BUS_ADDRESS" > /tmp/spotifyd_bus
echo "To use spotifyd's session bus, run 'export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/spotifyd_bus)'"
spotifyd --no-daemon --use-mpris
```

Run `dbus-launch ./spotify_wrapper.sh` and follow the instructions in the output.

**Option 2 – system bus**

Set `dbus_type = "system"` (or pass `--dbus-type system`) and allow your user to own the MPRIS name by creating `/usr/share/dbus-1/system.d/spotifyd.conf`:

```xml
<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-BUS Bus Configuration 1.0//EN" "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <policy user="youruser">
    <allow own_prefix="rs.spotifyd"/>
    <allow own_prefix="org.mpris.MediaPlayer2.spotifyd"/>
  </policy>
  <policy user="youruser">
    <allow send_destination_prefix="rs.spotifyd"/>
    <allow send_destination_prefix="org.mpris.MediaPlayer2.spotifyd"/>
  </policy>
</busconfig>
```

Reload D-Bus with `systemctl reload dbus` after creating the file. Ensure you are using a `spotifyd` build that includes the `dbus_mpris` feature (the `-full` binary or a custom build with `--features "dbus_mpris"`).

## Features

- Listens for track changes via MPRIS
- Captures audio through PipeWire/PulseAudio monitor sources
- Automatically detects sample rate and channel count
- Pauses recording when playback is paused
- Writes MP3 (or other formats) with ID3 tags and cover art
- Provides a Typer-based CLI with logging via Rich
- Automatically skips advertisements using track metadata
- Avoids re-recording tracks that already exist on disk
- Works with both Spotify and spotifyd via the configurable `--player` option
- Optionally creates an M3U playlist of recorded tracks using `--playlist <file>`
- Bundle playlist tracks into a single album using `--bundle-playlist`

## Usage

### With Wheel Installation
```bash
# Basic usage
spotify-splitter record

# Custom output directory and format
spotify-splitter --output ~/Music/Rips --format flac record

# Headless mode with spotifyd
spotify-splitter record --spotifyd-mode

# Save a playlist of recorded tracks
spotify-splitter record --playlist mysession.m3u
# Existing playlist files will be appended to rather than overwritten
# Bundle playlist tracks under a single album
spotify-splitter record --playlist mysession.m3u --bundle-playlist
```

### With Source Installation
```bash
# Basic usage
poetry run spotify-splitter record

# Custom output directory and format
poetry run spotify-splitter --output ~/Music/Rips --format flac record

# Headless mode with spotifyd
poetry run spotify-splitter record --spotifyd-mode

# Save a playlist of recorded tracks
poetry run spotify-splitter record --playlist mysession.m3u
# Existing playlist files will be appended to rather than overwritten
# Bundle playlist tracks under a single album
poetry run spotify-splitter record --playlist mysession.m3u --bundle-playlist
```

By default, tracks are saved under `~/Music/<Artist>/<Album>/<Artist> - <Title>.mp3`.

Use `--help` to view available options.

If you notice occasional "input overflow" warnings in the logs, try
increasing the audio buffer or adjusting the stream latency:

```bash
# With wheel installation
spotify-splitter record --queue-size 50 --latency 0.1

# With source installation
poetry run spotify-splitter record --queue-size 50 --latency 0.1
```

## ID3 Tagging API Integration

Spotify Splitter includes integration with an external ID3 tagging API service that automatically processes recorded tracks to enhance metadata tags with additional information like genre, year, and other missing tags.

### Automatic Processing on Shutdown

When recording ends, Spotify Splitter automatically calls the tagging API service to process your recorded tracks. The API runs at `http://localhost:5000` by default and provides two endpoints:

- `/process_directory` - Processes all MP3 files in a directory
- `/process_playlist` - Processes tracks listed in an M3U playlist file

### Setting Up the ID3 Tagger API

To use this feature, install and run the companion ID3 tagging service:

1. **Install the ID3 Tagger API**: [https://github.com/Brownster/id3-tagger](https://github.com/Brownster/id3-tagger)
2. **Start the service** on `http://localhost:5000`
3. **Record tracks** with Spotify Splitter - the API will be called automatically on shutdown

The tagging service enhances your tracks with:
- Genre information
- Release year
- Additional metadata from music databases
- Improved tag consistency

If the API service is not running, Spotify Splitter will continue to work normally and simply log a warning about the unavailable tagging service.


## Automated Library Management with Lidarr

You can also let [Lidarr](https://lidarr.audio/) handle tagging and
organization. Point Lidarr's "Manual Import" (Drone Factory) folder to the same
directory used by `spotify-splitter` and it will automatically match tracks and
move them into your music library.

For integration with Lidarr, configure Lidarr to monitor the same directory used by `spotify-splitter`. When new files appear, Lidarr will import them, fetch metadata, and move them into your organized library.


# Not using any post processing adding music folder to Strawberry
<img width="400" height="764" alt="image" src="https://github.com/user-attachments/assets/e2cebd73-57da-4d80-a896-90ffa5c6c804" />



## Troubleshooting

If you see an error like `ValueError: No input device matching` when starting a
recording, the monitor name reported by `pactl` might not match the name used by
PortAudio. The application now attempts a best-effort lookup but you may need to
ensure the correct audio backend and monitor sources are available.
If saved tracks sound distorted or play at the wrong speed, the capture
device's sample rate likely doesn't match Spotify's output. Verify that the
selected monitor source uses the same sample rate reported by `pactl`.


## Technical Overview

### Architecture and Design

Spotify Splitter is designed to solve the challenge of automated music archival from streaming services. The application leverages several key technologies to achieve reliable, high-quality audio capture and track segmentation on Linux systems.

### Core Components

**MPRIS Integration**
The application utilizes the Media Player Remote Interfacing Specification (MPRIS) through D-Bus to monitor track metadata from Spotify clients. This standardized interface provides real-time access to track information including artist, title, album, and artwork URLs, enabling precise track boundaries and metadata tagging.

**Audio Capture System**
Audio capture is implemented using PulseAudio/PipeWire monitor sources, which provide a lossless digital copy of the audio stream. The system employs the sounddevice library with PortAudio backend for consistent cross-platform audio handling, with automatic sample rate detection and buffer management to prevent audio dropouts.

**Signal Processing Pipeline**
The audio processing pipeline handles format conversion from float32 to int16, implements configurable buffering strategies, and provides real-time audio level monitoring. The system includes overflow protection and dynamic latency adjustment to maintain audio quality under varying system loads.

### Technical Challenges and Solutions

**Data Type Compatibility**
Early development revealed audio format mismatches between capture and export libraries. The solution involved implementing proper format conversion with scaling and type casting to ensure compatibility between sounddevice's float32 output and pydub's integer-based processing.

**Track Boundary Detection**
Accurate track splitting required developing a timestamp-based segmentation system that monitors MPRIS metadata changes while maintaining audio continuity. The system includes validation logic to ensure complete track capture and duration verification against expected track lengths.

**System Integration**
The application handles the complexity of Linux audio systems by implementing automatic device detection for both PulseAudio and PipeWire environments. This includes dynamic monitor source discovery and sample rate adaptation for different audio configurations.

### Production Considerations

**Reliability Features**
- Incomplete track detection prevents partial file saves
- Advertisement filtering using track metadata analysis
- Duplicate detection to avoid re-recording existing tracks
- Automatic cache management for long-running sessions

**Performance Optimizations**
- Configurable buffer sizes for different system capabilities
- Threaded audio processing to prevent blocking
- Memory-efficient streaming for extended recording sessions
- Adaptive latency control for various hardware configurations

**Headless Operation**
Integration with spotifyd enables automated recording for server deployments, with optimized buffer management and error recovery suitable for unattended operation.

### Technical Stack

Built on modern Python libraries including sounddevice for audio capture, pydbus for D-Bus communication, mutagen for metadata handling, and rich for user interface components. The application follows modern software engineering practices with comprehensive testing, type hints, and modular architecture.

This tool represents a practical solution to digital music archival challenges, combining robust audio processing with reliable metadata handling to create a professional-grade recording system for personal use.
