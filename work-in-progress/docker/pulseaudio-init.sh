#!/bin/bash
set -e

echo "Initializing PulseAudio..."

# Set up runtime directory
export XDG_RUNTIME_DIR="/run/user/1000"
export PULSE_RUNTIME_PATH="$XDG_RUNTIME_DIR/pulse"

mkdir -p "$XDG_RUNTIME_DIR/pulse"

# Create PulseAudio config
mkdir -p /home/spotify/.config/pulse

cat > /home/spotify/.config/pulse/client.conf << EOF
# PulseAudio client configuration for Spotify Splitter
autospawn = yes
default-server = unix:$PULSE_RUNTIME_PATH/native
EOF

cat > /home/spotify/.config/pulse/daemon.conf << EOF
# PulseAudio daemon configuration for Spotify Splitter
daemonize = no
fail = yes
high-priority = yes
nice-level = -11
realtime-scheduling = yes
realtime-priority = 5
resample-method = speex-float-1
default-sample-format = float32le
default-sample-rate = 44100
default-sample-channels = 2
default-channel-map = front-left,front-right
EOF

cat > /home/spotify/.config/pulse/default.pa << EOF
#!/usr/bin/pulseaudio -nF

# Load necessary modules
load-module module-native-protocol-unix auth-anonymous=1 socket=$PULSE_RUNTIME_PATH/native
load-module module-null-sink sink_name=spotify_output sink_properties=device.description="Spotify-Output"
load-module module-virtual-source source_name=spotify_monitor master=spotify_output.monitor source_properties=device.description="Spotify-Monitor"

# Set the default sink and source
set-default-sink spotify_output
set-default-source spotify_monitor
EOF

# Start PulseAudio
echo "Starting PulseAudio daemon..."
exec pulseaudio --verbose --log-target=stderr --system=false --daemonize=false