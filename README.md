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
- Works with both Spotify and spotifyd via the configurable `--player` option

## Usage

```bash
poetry run spotify-splitter record
```

By default, tracks are saved under `~/Music/<Artist>/<Album>/<Artist> - <Title>.mp3`.

For a custom output directory and format:

```bash
poetry run spotify-splitter --output ~/Music/Rips --format flac record
```

To use the headless `spotifyd` client instead of the desktop app:

```bash
poetry run spotify-splitter record --player spotifyd
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


From Clicks to Code: Building a Headless Spotify Recorder

This project started with a simple, almost nostalgic idea: what if I could have a personal, offline copy of my Spotify playlists, just like the old days of curating an MP3 library? The manual process of recording and splitting tracks is tedious and error-prone. I wanted a tool that could do it for me, automatically and reliably. This is the story of building that tool, spotify-splitter.
The Core Idea: Listen, Record, Split

The initial concept was straightforward:

    Listen for what song is currently playing on Spotify.

    Record the system's audio output.

    Split the recording into a new file every time the song changes.

    Tag the file with the correct artist, title, and album information.

This simple idea, however, led to a fascinating journey through the layers of the Linux desktop stack.
Getting the Data: MPRIS and D-Bus

The first challenge was getting the track information. How does a script know what Spotify is playing? The answer is MPRIS (Media Player Remote Interfacing Specification), a standard D-Bus interface. Most media players on Linux, including Spotify and spotifyd, use it to publish "Now Playing" information and accept commands like play, pause, and skip.

Using the pydbus library, the script connects to the user's session D-Bus and listens for property changes from the org.mpris.MediaPlayer2.spotify service. This provides all the essential metadata: artist, title, album, and even a URL for the album art.
Capturing the Audio: The Invisible Microphone

With the metadata sorted, the next step was to capture the audio. The most robust way to do this on a modern Linux system is to record from a monitor source. A monitor source is a virtual input device that mirrors a real output device. In simple terms, it's an invisible microphone listening directly to what your speakers or headphones are playing.

The sounddevice library proved perfect for this. It provides a clean interface to the underlying PortAudio library. By using pactl (the PulseAudio/PipeWire command-line tool) to find the name of Spotify's monitor source, we could tell sounddevice exactly which stream to record.
The First Major Bug: The Wall of Noise

The initial recordings were a success... in that they created files. The audio itself was a horrifying, distorted mess of static. After some excellent debugging help, the culprit was found: a fundamental data type mismatch.

    sounddevice was capturing high-precision float32 audio samples (where values range from -1.0 to 1.0).

    pydub, the library used for exporting the MP3, was being told to interpret the raw bytes of those floats as if they were 32-bit integers.

The solution was to convert the audio format before exporting. By scaling the float values and converting them to standard 16-bit integers, the garbled noise was transformed into crystal-clear audio.
Generated python

      
# The key to clean audio: converting from float to int16
raw_float_samples = np.concatenate(self.buffer)
int_samples = (raw_float_samples * np.iinfo(np.int16).max).astype(np.int16)

# Now, pydub gets the data it expects
audio_segment = AudioSegment(
    int_samples.tobytes(),
    sample_width=2, # 16-bit = 2 bytes
    ...
)

    

IGNORE_WHEN_COPYING_START
Use code with caution. Python
IGNORE_WHEN_COPYING_END
Making It Robust: From Script to Tool

With the core functionality working, the focus shifted to making the tool truly reliable for long-term, headless use. This involved adding several key features:

    Handling Incomplete Tracks: A track is only saved if the script has seen it from beginning to end. This prevents saving partial rips if the script is started mid-song or is stopped before a track finishes.

    Skipping Ads and Existing Files: The script inspects the track ID to differentiate songs from ads and checks if a file already exists to avoid re-recording entire playlists.

    Automatic Device Detection: The code was made more resilient to handle the different ways that PipeWire and PulseAudio name their audio devices, and to dynamically detect the correct sample rate.

    A Polished UI: Using rich, the application now provides a live-updating spinner that gives the user constant feedback on what's happening, from recording to pausing to skipping ads.

The Headless Dream: Docker and spotifyd

The final goal was to run this as a completely automated service. The solution was to combine spotify-splitter with spotifyd (a headless Spotify client) and package everything in Docker.

A docker-compose.yml file now orchestrates the entire setup:

    The spotifyd Service: Runs the headless Spotify client, which plays audio to the system.

    The spotify-splitter Service: Runs our application, which finds the audio from spotifyd and records it.

This setup is perfect for a home server or a Raspberry Pi. You can queue up a massive playlist from your phone, tell it to play on the headless client, and walk away. Hours later, you'll have a folder full of perfectly recorded and split tracks.
Conclusion

spotify-splitter is a testament to the power of the open-source ecosystem. It stands on the shoulders of giants, from the Linux audio stack to the developers of mutagen, pydub, and rich. It began as a simple idea and evolved through a challenging but rewarding debugging process into a stable, feature-rich tool that solves a real-world problem.

I hope you find it as useful as I do. Happy listening, and happy hacking
