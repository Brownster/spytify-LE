# Spotify Splitter

This project records Spotify playback on Linux and saves each track as an individual audio file with metadata.

<img width="1508" height="717" alt="image" src="https://github.com/user-attachments/assets/8b02dc7c-4ef2-4916-8c4b-95ce27e7e2b6" />


## Prerequisites

- Linux with PulseAudio or PipeWire
- A running Spotify client
- [Poetry](https://python-poetry.org/) installed
- Python 3.10– <4.0
- `python3-pyaudio` installed (system package)
- `ffmpeg` installed (system package)

ffmpeg is required for converting audio during export and when running tests.

## Installation

```bash
git clone https://github.com/Brownster/spoti2.git
cd spoti2
sudo apt install python3-pyaudio
poetry config virtualenvs.options.system-site-packages true
poetry install
```

### Docker Compose

Alternatively, you can run the project using Docker Compose. This is recommended
for headless operation.

1. **Clone the repository**
   ```bash
   git clone https://github.com/Brownster/spoti2.git
   cd spoti2
   ```
2. **Configure `spotifyd`**
   Edit the `spotifyd.conf` file and enter your Spotify Premium credentials.
3. **Configure your music directory**
   Open the `.env` file and set `MUSIC_PATH` to the folder where ripped tracks
   should be saved. By default this points to a `Music` directory inside the
   project.
4. **Configure Beets (optional)**
   The container can automatically run `beet import -AW /Music` on a schedule.
   Set `BEETS_CRON_SCHEDULE` in `.env` to control how often the import runs and
   edit files under `./beets` to customise the configuration. If you plan to use
   Lidarr instead, set `INSTALL_BEETS=false` in `.env` and remove the Beets
   volume mapping from the Compose file before building.
5. **Expose your audio devices**
   Both containers need access to the host PulseAudio or PipeWire socket in
   order to play and capture audio. Mount `/run/user/1000/pulse` (or the
   appropriate path for your user) and `/dev/snd` into the containers.
   Example volume mappings are shown below.
6. **Run the services**
   ```bash
   docker-compose up --build -d
   ```

Recorded tracks will be saved to the path specified by `MUSIC_PATH`.

## Features

- Listens for track changes via MPRIS
- Captures audio through PipeWire/PulseAudio monitor sources
- Automatically detects sample rate and channel count
- Pauses recording when playback is paused
- Writes MP3 (or other formats) with ID3 tags and cover art
- Provides a Typer-based CLI with logging via Rich
- Automatically skips advertisements using track metadata
- Avoids re-recording tracks that already exist on disk

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

## Post-Processing with Beets (Optional we do add enough tag info to be imported ok into IPOD / MP3 player, Year and Genre will not be adeded by spoti2)

This tool is designed to produce raw track rips. For the best results,
install [Beets](https://beets.io/) to automatically tag and organize your
music library.

### 1. Install Beets

```bash
pip install beets
```

### 2. Configure Beets

Create a configuration file at `~/.config/beets/config.yaml` with the
following content to enable album art fetching and define your music
directory:

```yaml
# ~/.config/beets/config.yaml
directory: ~/Music
library: ~/.config/beets/musiclibrary.db

plugins: fetchart

fetchart:
    cautious: true
    sources:
      - coverartarchive
      - albumartexchange
      - amazon
```

### 3. Rip and Import

With Beets configured, your workflow becomes two steps:

1. Record tracks using Spotify Splitter, specifying a separate output
   directory for the raw rips:

   ```bash
   poetry run spotify-splitter --output ~/Music/SpotifyRips record
   ```

2. Import the new rips with Beets:

   ```bash
   beet import -i ~/Music/SpotifyRips
   ```

Beets will fetch metadata and artwork and move the files into your main
`~/Music` library, asking you to confirm matches along the way.

### Import Without Moving (Optional)

If you want to keep the default `~/Music` location and only tag files in
place, run:

```bash
beet import -AW ~/Music
```

The `-A` flag avoids copying or moving files while `-W` writes the updated
metadata. To skip files you've already imported, enable incremental mode in
your configuration:

```yaml
import:
  incremental: true
```

## Automated Library Management with Lidarr

You can also let [Lidarr](https://lidarr.audio/) handle tagging and
organization. Point Lidarr's "Manual Import" (Drone Factory) folder to the same
directory used by `spotify-splitter` and it will automatically match tracks and
move them into your music library.

### Example Docker Compose

```yaml
version: '3.8'
services:
  lidarr:
    image: lscr.io/linuxserver/lidarr:latest
    container_name: lidarr
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=Europe/London
    volumes:
      - ./lidarr_config:/config
      - /path/on/host/to/your/music:/music
      - /path/on/host/to/downloads:/downloads
    ports:
      - 8686:8686
    restart: unless-stopped

  spotifyd:
    image: spotifyd/spotifyd:latest
    network_mode: "host"
    volumes:
      - ./spotifyd.conf:/etc/spotifyd.conf
      - spotifyd-cache:/var/cache/spotifyd
      - /run/user/1000/pulse:/run/user/1000/pulse
      - /dev/snd:/dev/snd
    restart: unless-stopped

  spotify-splitter:
    build:
      context: .
      args:
        INSTALL_BEETS: ${INSTALL_BEETS}
    container_name: spotify-splitter
    network_mode: "host"
    depends_on:
      - spotifyd
    env_file:
      - .env
    environment:
      INSTALL_BEETS: ${INSTALL_BEETS}
    volumes:
      - "${MUSIC_PATH}:/Music"
      - "${BEETS_CONFIG_PATH}:/root/.config/beets"
      - /run/user/1000/pulse:/run/user/1000/pulse
      - /dev/snd:/dev/snd
    restart: unless-stopped
```

A complete Compose file integrating Spotify Splitter with a typical *arr stack
can be found at `docs/arr-example-dockerfile`.

Set `OUTPUT_DIR` inside `spotify-splitter` to `/downloads/spotify_rips` and
configure Lidarr to monitor the same path. When new files appear, Lidarr will
import them, fetch metadata, and move them into your organized library.


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
