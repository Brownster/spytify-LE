#!/bin/bash
set -e

echo "Starting Spotify Splitter Standalone Container..."

# Validate required environment variables
if [ -z "$SPOTIFY_USERNAME" ] || [ -z "$SPOTIFY_PASSWORD" ]; then
    echo "ERROR: SPOTIFY_USERNAME and SPOTIFY_PASSWORD must be set"
    echo "Example: docker run -e SPOTIFY_USERNAME=your_username -e SPOTIFY_PASSWORD=your_password ..."
    exit 1
fi

# Create spotifyd config from template if it doesn't exist
if [ ! -f "/config/spotifyd.conf" ]; then
    echo "Creating spotifyd configuration..."
    sed -e "s/YOUR_SPOTIFY_USERNAME/$SPOTIFY_USERNAME/g" \
        -e "s/YOUR_SPOTIFY_PASSWORD/$SPOTIFY_PASSWORD/g" \
        -e "s/YOUR_DEVICE_NAME/$SPOTIFY_DEVICE_NAME/g" \
        -e "s/YOUR_BITRATE/$SPOTIFY_BITRATE/g" \
        /app/spotifyd-template.conf > /config/spotifyd.conf
    chown spotify:spotify /config/spotifyd.conf
    echo "spotifyd.conf created at /config/spotifyd.conf"
fi

# Ensure proper permissions
chown -R spotify:spotify /config /Music /var/log/spotify-splitter

# Initialize D-Bus for user session
mkdir -p /run/user/1000
chown spotify:spotify /run/user/1000

# Set audio environment variables
export PULSE_RUNTIME_PATH="/run/user/1000/pulse"
export XDG_RUNTIME_DIR="/run/user/1000"

# Create D-Bus directories and socket
mkdir -p /run/dbus /var/run/dbus
chmod 755 /run/dbus /var/run/dbus

# Start D-Bus system service
echo "Starting D-Bus system service..."
# Initialize D-Bus system configuration
dbus-uuidgen --ensure=/etc/machine-id
dbus-daemon --system --fork --pid=/var/run/dbus/pid

echo "Starting services with supervisor..."
echo "  - PulseAudio"
echo "  - spotifyd (device: $SPOTIFY_DEVICE_NAME)"
echo "  - spotify-splitter (output: /Music)"

# Start supervisor to manage all services
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf