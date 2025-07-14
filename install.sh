#!/bin/bash
set -e

echo "🎵 Spotify Splitter Installation"
echo "================================"

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This tool only works on Linux"
    exit 1
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y python3-pip python3-pyaudio ffmpeg
elif command -v dnf &> /dev/null; then
    sudo dnf install -y python3-pip python3-pyaudio ffmpeg
elif command -v pacman &> /dev/null; then
    sudo pacman -S python-pip python-pyaudio ffmpeg
else
    echo "⚠️  Please install python3-pip, python3-pyaudio, and ffmpeg manually"
fi

# Install spotify-splitter
echo "🔧 Installing spotify-splitter..."
if [ -f "spotify_splitter-0.1.0-py3-none-any.whl" ]; then
    pip install --user spotify_splitter-0.1.0-py3-none-any.whl
else
    pip install --user spotify-splitter
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Usage:"
echo "  spotify-splitter --output ~/Music record"
echo "  spotify-splitter record --spotifyd-mode  # For headless usage"
echo ""
echo "📚 For more information: https://github.com/Brownster/spoti2"