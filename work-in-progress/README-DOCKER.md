# Spotify Splitter Docker Setup

## Standalone Container (Recommended)

The standalone container includes both spotifyd and spotify-splitter in a single container.

### Quick Start

1. **Copy the environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your Spotify credentials:**
   ```bash
   SPOTIFY_USERNAME=your_username
   SPOTIFY_PASSWORD=your_password
   SPOTIFY_DEVICE_NAME=Spotify-Headless
   MUSIC_PATH=./Music
   CONFIG_PATH=./config
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose -f docker-compose.standalone.yml up -d
   ```

### Manual Docker Run

```bash
docker build -f Dockerfile.standalone -t spotify-splitter-standalone .

docker run -d \
  --name spotify-splitter \
  --restart unless-stopped \
  --device /dev/snd:/dev/snd \
  --cap-add SYS_NICE \
  --cap-add SYS_RESOURCE \
  --network host \
  -e SPOTIFY_USERNAME="your_username" \
  -e SPOTIFY_PASSWORD="your_password" \
  -e SPOTIFY_DEVICE_NAME="Spotify-Headless" \
  -v $(pwd)/Music:/Music \
  -v $(pwd)/config:/config \
  -v $(pwd)/logs:/var/log/spotify-splitter \
  spotify-splitter-standalone
```

## What You Need to Pass Through

### Essential Requirements

1. **Audio Device Access:**
   ```bash
   --device /dev/snd:/dev/snd
   ```

2. **Audio Capabilities:**
   ```bash
   --cap-add SYS_NICE
   --cap-add SYS_RESOURCE
   ```

3. **Network Access:**
   ```bash
   --network host  # For MPRIS/D-Bus communication
   ```

4. **Spotify Credentials:**
   ```bash
   -e SPOTIFY_USERNAME="your_username"
   -e SPOTIFY_PASSWORD="your_password"
   ```

5. **Volume Mounts:**
   ```bash
   -v /path/to/music:/Music           # Output directory
   -v /path/to/config:/config         # Configuration files
   -v /path/to/logs:/var/log/spotify-splitter  # Logs (optional)
   ```

### Optional Environment Variables

- `SPOTIFY_DEVICE_NAME` - Name shown in Spotify Connect (default: "Spotify-Headless")
- `SPOTIFY_BITRATE` - Audio quality: 96, 160, 320 (default: 320)
- `VERBOSE` - Enable verbose logging (default: false)

### Manual Usage with Spotifyd

Spotifyd streams are now **automatically detected** via PulseAudio/PipeWire! The app will find spotifyd streams using these identifiers:
- `application.name = "librespot"`
- `application.process.binary = "spotifyd"`
- `media.name = "Spotify endpoint"`

For optimal spotifyd performance, use the `--spotifyd-mode` flag:

```bash
# With existing spotifyd instance (auto-detects stream)
poetry run spotify-splitter --spotifyd-mode record

# Or specify custom player name for MPRIS
poetry run spotify-splitter --player spotifyd record
```

**Spotifyd mode optimizations:**
- Automatically uses "spotifyd" as MPRIS player name
- Increases buffer sizes (300+ blocks minimum)  
- Uses larger block sizes (4096 frames) for stability
- Sets higher latency tolerance for headless environments
- **Audio stream auto-detection** works with both Spotify client and spotifyd

## Directory Structure

After running, you'll have:

```
./
├── Music/           # Recorded tracks (mapped to /Music in container)
├── config/          # Configuration files (mapped to /config in container)
│   └── spotifyd.conf    # Auto-generated on first run
├── logs/            # Application logs (optional)
└── .env             # Environment variables
```

## How It Works

1. **Container starts** and creates spotifyd.conf from your credentials
2. **PulseAudio** initializes with virtual audio devices
3. **spotifyd** starts and appears as "Spotify-Headless" in Spotify Connect
4. **spotify-splitter** starts in spotifyd mode with optimizations:
   - Uses "spotifyd" as MPRIS player name
   - Larger audio buffers (300+ blocks) for stability
   - Higher latency tolerance for headless operation
   - Larger block sizes (4096 frames) for better performance
5. **Tracks are saved** to the mounted /Music directory

## Monitoring

**View logs:**
```bash
docker logs spotify-splitter-standalone
```

**Check service status:**
```bash
docker exec spotify-splitter-standalone supervisorctl status
```

**Health check:**
```bash
docker inspect spotify-splitter-standalone | grep -A 5 Health
```

## Troubleshooting

**No audio devices:**
- Ensure `/dev/snd` exists on host
- Check audio capabilities are added
- Verify user has audio permissions on host

**Spotify device not appearing:**
- Check credentials in .env file
- Verify network connectivity
- Check spotifyd logs: `docker logs spotify-splitter-standalone`

**No tracks being saved:**
- Verify /Music volume mount
- Check permissions on music directory
- Ensure Spotify is playing on the device

**MPRIS connection issues:**
- Ensure network_mode: host is set
- Check D-Bus is running in container
- Verify spotifyd is using MPRIS