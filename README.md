# Spotify Splitter

This project records Spotify playback on Linux and saves each track as an individual audio file with metadata.

## Features

- Listens for track changes via MPRIS
- Captures audio through PipeWire/PulseAudio monitor sources
- Writes MP3 (or other formats) with ID3 tags and cover art
- Provides a Typer-based CLI with logging via Rich

## Usage

```bash
poetry run spotify-splitter record
```

Use `--help` to view available options.
