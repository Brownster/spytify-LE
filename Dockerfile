# Use a specific Python version that you know works
FROM python:3.12-slim

# Install system dependencies needed by this script and spotifyd
# `portaudio19-dev` is needed by sounddevice/pyaudio
# `pulseaudio-utils` contains `pactl`
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    pulseaudio-utils \
    git \
    cron \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Optionally install Beets
ARG INSTALL_BEETS=true
RUN if [ "$INSTALL_BEETS" = "true" ]; then \
        pip install poetry beets; \
    else \
        pip install poetry; \
    fi

# Set up the work directory
WORKDIR /app

# Copy just the files needed to install dependencies
COPY poetry.lock pyproject.toml ./

# Install Python dependencies into a virtual environment
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-dev

# Copy the rest of your application code
COPY . .

# Add entrypoint script
RUN chmod +x /app/start.sh

# This is the command that will run when the container starts
CMD ["/app/start.sh"]
