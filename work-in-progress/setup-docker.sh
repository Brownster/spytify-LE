#!/bin/bash
set -e

echo "🎵 Spotify Splitter Docker Setup"
echo "================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Starting Docker..."
    sudo systemctl start docker
    sleep 2
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p ~/Music config logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your Spotify credentials before continuing!"
    echo "   SPOTIFY_USERNAME=your_username"
    echo "   SPOTIFY_PASSWORD=your_password"
    echo ""
    read -p "Press Enter after editing .env file..."
fi

# Build the container
echo "🔨 Building Docker container..."
docker build -f Dockerfile.standalone -t spotify-splitter-standalone .

# Check if container is already running
if docker ps | grep -q spotify-splitter; then
    echo "🔄 Stopping existing container..."
    docker stop spotify-splitter
    docker rm spotify-splitter
fi

# Run the container
echo "🚀 Starting Spotify Splitter..."
docker run -d \
  --name spotify-splitter \
  --restart unless-stopped \
  --device /dev/snd:/dev/snd \
  --cap-add SYS_NICE \
  --cap-add SYS_RESOURCE \
  --network host \
  --env-file .env \
  -v ~/Music:/Music \
  -v $(pwd)/config:/config \
  -v $(pwd)/logs:/var/log/spotify-splitter \
  spotify-splitter-standalone

echo ""
echo "✅ Spotify Splitter is running!"
echo ""
echo "📊 View logs:"
echo "   docker logs -f spotify-splitter"
echo ""
echo "🎯 Connect to Spotify:"
echo "   1. Open Spotify on your phone/computer"
echo "   2. Start playing music"
echo "   3. Click the Spotify Connect icon"
echo "   4. Select 'Spotify-Headless' device"
echo "   5. Tracks will be saved to ~/Music"
echo ""
echo "🛑 Stop container:"
echo "   docker stop spotify-splitter"