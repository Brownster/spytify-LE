# Installation Guide

Spytify-LE is a Linux Spotify desktop recorder. The recommended install path is
`pipx`, because it gives the CLI its own Python environment while keeping the
`spotify-splitter` command on your `PATH`.

## Fastest Install

From a cloned checkout:

```bash
./install.sh
```

The script installs distro packages for:

- `pipx`
- PyGObject / `python3-gi` for MPRIS/D-Bus
- PortAudio for audio capture
- `ffmpeg` for export

Then it installs the current checkout with:

```bash
pipx install --force --system-site-packages .
```

`--system-site-packages` is intentional: the recorder needs the distro-provided
PyGObject bindings used by `gi.repository`.

## Manual Install

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y pipx python3-gi gir1.2-glib-2.0 libportaudio2 ffmpeg
pipx ensurepath
pipx install --system-site-packages 'git+https://github.com/Brownster/spytify-LE.git'
```

Fedora:

```bash
sudo dnf install -y pipx python3-gobject portaudio ffmpeg
pipx ensurepath
pipx install --system-site-packages 'git+https://github.com/Brownster/spytify-LE.git'
```

Arch:

```bash
sudo pacman -S --needed python-pipx python-gobject portaudio ffmpeg
pipx ensurepath
pipx install --system-site-packages 'git+https://github.com/Brownster/spytify-LE.git'
```

## First Run

Open Spotify, start playing a track, then run:

```bash
spotify-splitter doctor
spotify-splitter web
```

The web UI opens on `http://127.0.0.1:8730` and shows readiness checks before you
start recording.

## Update

```bash
pipx upgrade spotify-splitter
```

For a local checkout:

```bash
git pull
pipx install --force --system-site-packages .
```

## Uninstall

```bash
pipx uninstall spotify-splitter
rm -rf ~/.config/spotify_splitter ~/.cache/spotify_splitter
```
