# Use a specific Python version that you know works
FROM python:3.12-slim

# Install system dependencies needed by your script and spotifyd
# `portaudio19-dev` is needed by sounddevice/pyaudio
# `pulseaudio-utils` contains `pactl`
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    pulseaudio-utils \
    git \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set up the work directory
WORKDIR /app

# Copy just the files needed to install dependencies
COPY poetry.lock pyproject.toml ./

# Install Python dependencies into a virtual environment
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-dev

# Copy the rest of your application code
COPY . .

# This is the command that will run when the container starts
CMD ["poetry", "run", "spotify-splitter", "record"]
