#!/bin/sh
set -e

if [ "${INSTALL_BEETS:-true}" = "true" ]; then
    CRON_SCHEDULE="${BEETS_CRON_SCHEDULE:-0 */6 * * *}"
    CONFIG_DIR="${BEETS_CONFIG_DIR:-/root/.config/beets}"
    mkdir -p "$CONFIG_DIR"

    echo "$CRON_SCHEDULE beet -c $CONFIG_DIR/config.yaml import -AW /Music >> /var/log/beets.log 2>&1" > /etc/cron.d/beets
    chmod 0644 /etc/cron.d/beets
    crontab /etc/cron.d/beets
    cron
fi

exec poetry run spotify-splitter --output /Music --spotifyd-mode record
