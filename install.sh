#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_URL="git+https://github.com/Brownster/spytify-LE.git"

fct_install_system_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y pipx python3-gi gir1.2-glib-2.0 libportaudio2 ffmpeg
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y pipx python3-gobject portaudio ffmpeg
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed python-pipx python-gobject portaudio ffmpeg
  else
    printf '%s\n' "Unsupported package manager."
    printf '%s\n' "Install pipx, PyGObject/python3-gi, PortAudio, and ffmpeg manually."
  fi
}

fct_install_app() {
  if ! command -v pipx >/dev/null 2>&1; then
    printf '%s\n' "pipx was not found after installing dependencies."
    printf '%s\n' "Install pipx, then rerun this script."
    exit 1
  fi

  pipx ensurepath

  if [[ -f "pyproject.toml" && -d "spotify_splitter" ]]; then
    pipx install --force --system-site-packages .
  else
    pipx install --force --system-site-packages "${REPO_URL}"
  fi
}

fct_print_next_steps() {
  cat <<'EOF'

Installation complete.

If this is a new shell, restart it or run:
  export PATH="$HOME/.local/bin:$PATH"

Then run:
  spotify-splitter doctor
  spotify-splitter web
EOF
}

fct_main() {
  if [[ "${OSTYPE:-}" != linux* ]]; then
    printf '%s\n' "spotify-splitter only supports Linux."
    exit 1
  fi

  fct_install_system_deps
  fct_install_app
  fct_print_next_steps
}

fct_main "$@"
