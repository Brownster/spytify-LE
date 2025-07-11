# Spotify Splitter

This project records Spotify playback on Linux and saves each track as an individual audio file with metadata.

## Prerequisites

- Linux with PulseAudio or PipeWire
- A running Spotify client
- [Poetry](https://python-poetry.org/) installed

## Installation

```bash
git clone https://example.com/spotify-splitter.git
cd spotify-splitter
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

For a custom output directory and format:

```bash
poetry run spotify-splitter --output ~/Music/Rips --format flac record
```

Use `--help` to view available options.
