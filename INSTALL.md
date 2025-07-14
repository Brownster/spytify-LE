# Installation Guide

## Quick Install

### Method 1: Using the install script (recommended)
```bash
curl -O https://github.com/Brownster/spoti2/releases/latest/download/install.sh
chmod +x install.sh
./install.sh
```

### Method 2: Manual installation
1. **Install system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt install python3-pip python3-pyaudio ffmpeg
   
   # Fedora
   sudo dnf install python3-pip python3-pyaudio ffmpeg
   ```

2. **Install spotify-splitter:**
   ```bash
   pip install --user spotify_splitter-0.1.0-py3-none-any.whl
   ```

3. **Add to PATH** (if needed):
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Usage

### Basic Usage
```bash
# Record tracks to ~/Music
spotify-splitter record

# Custom output directory
spotify-splitter --output ~/Music/Spotify record

# Different format
spotify-splitter --format flac record
```

### Headless Usage with spotifyd
```bash
# Install and configure spotifyd first
sudo dnf install spotifyd  # or from GitHub releases

# Configure ~/.config/spotifyd/spotifyd.conf
[global]
username = "your_username"
password = "your_password"  
device_name = "MyDevice"
backend = "pulseaudio"
bitrate = 320

# Start spotifyd
systemctl --user start spotifyd

# Run spotify-splitter in spotifyd mode
spotify-splitter record --spotifyd-mode
```

## Troubleshooting

### Common Issues
- **"No input device"**: Make sure Spotify is playing and audio is working
- **"Permission denied"**: Try `pip install --user` instead of system-wide
- **"Command not found"**: Add `~/.local/bin` to your PATH

### Getting Help
- Check the [README](https://github.com/Brownster/spoti2) for detailed documentation
- Report issues at https://github.com/Brownster/spoti2/issues