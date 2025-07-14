# Developer's Guide

Welcome, developer! Thank you for your interest in contributing to Spotify Splitter. This guide walks you through the architecture and the recommended workflow for hacking on the project.

## Project Philosophy

The application aims to be a robust, fire-and-forget tool for recording Spotify streams. Because the audio and metadata systems operate independently, the code is structured to tolerate timing mismatches between them.

We coordinate three separate systems:

1. **MPRIS (D-Bus)** – reports metadata events when tracks change.
2. **PulseAudio/PipeWire** – provides the audio stream.
3. **The Python application** – ties everything together.

## Core Concepts

### The "Tape Recorder" Pipeline

Processing is split into two stages connected by queues:

1. **Collectors (main thread)**
   - `AudioStream` captures raw audio frames in a high-priority callback and immediately places them on an `audio_queue`.
   - `track_events` listens for MPRIS signals and pushes markers onto an `event_queue`.

2. **Processor (background thread)**
   - `SegmentManager` runs in its own thread. It ingests audio from `audio_queue` into a continuous buffer and checks `event_queue` for new track markers.
   - When it has enough markers for a complete song, it splits and saves the track.

### Smart Split Hierarchy

`SegmentManager.process_segments()` uses a three-tiered approach to handle desynchronization or mid-song pauses:

1. **Silence detection** – split on silence using `pydub` and verify the result against the metadata track length.
2. **Targeted search** – if the first pass fails, search for the first long silence near the expected end of the track.
3. **Failsafe hard cut** – if all else fails, save the entire block of audio between markers.

## Codebase Tour

- **`spotify_splitter/main.py`** – CLI entry point. Sets up logging, queues and threads.
- **`spotify_splitter/mpris.py`** – Handles D-Bus metadata and posts track events.
- **`spotify_splitter/audio.py`** – Captures audio using `sounddevice`.
- **`spotify_splitter/segmenter.py`** – Implements the processing loop and exporting logic.
- **`spotify_splitter/util.py`** – Helper functions for audio device detection.
- **`tests/`** – Unit tests that mock heavy dependencies so they run quickly.

## Development Environment

```bash
# Clone and enter the repo
git clone https://github.com/Brownster/spoti2.git
cd spoti2

# Install system packages (example for Debian/Ubuntu)
sudo apt-get update
sudo apt-get install python3-gi python3-pyaudio ffmpeg

# Allow Poetry venv to access system packages
poetry config virtualenvs.options.system-site-packages true

# Create the venv and install Python deps
poetry env use python3.12
poetry install
```

Run the tests with:

```bash
poetry run pytest
```

## Contributing

1. Open an issue to discuss your idea or bug fix.
2. Fork the repo and create a branch for your change.
3. Add or update tests where appropriate.
4. Ensure `poetry run pytest` passes.
5. Open a pull request linking to your issue.

Thank you for helping improve Spotify Splitter!
