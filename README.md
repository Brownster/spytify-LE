# Spotify Splitter

This project records Spotify playback on Linux and saves each track as an individual audio file with metadata.

## Prerequisites

- Linux with PulseAudio or PipeWire
- A running Spotify client
- [Poetry](https://python-poetry.org/) installed
- Python 3.10– <4.0
- `python3-pyaudio` installed (system package)

## Installation

```bash
git clone https://example.com/spotify-splitter.git
cd spotify-splitter
sudo apt install python3-pyaudio
poetry config virtualenvs.options.system-site-packages true
poetry install
```

## Features

- Listens for track changes via MPRIS
- Captures audio through PipeWire/PulseAudio monitor sources
- Automatically detects sample rate and channel count
- Pauses recording when playback is paused
- Writes MP3 (or other formats) with ID3 tags and cover art
- Provides a Typer-based CLI with logging via Rich
- Automatically skips advertisements using track metadata

## Usage

```bash
poetry run spotify-splitter record
```

By default, tracks are saved under `~/Music/<Artist>/<Album>/<Artist> - <Title>.mp3`.

For a custom output directory and format:

```bash
poetry run spotify-splitter --output ~/Music/Rips --format flac record
```

Use `--help` to view available options.

## Troubleshooting

If you see an error like `ValueError: No input device matching` when starting a
recording, the monitor name reported by `pactl` might not match the name used by
PortAudio. The application now attempts a best-effort lookup but you may need to
ensure the correct audio backend and monitor sources are available.
