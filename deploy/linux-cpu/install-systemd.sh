#!/usr/bin/env bash
# Install systemd unit for alpr on Linux CPU.
# Usage: sudo bash deploy/linux-cpu/install-systemd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
USER_NAME="${SUDO_USER:-ubuntu}"
HOME_DIR="$(eval echo "~${USER_NAME}")"
# Prefer project path under the invoking user's home if ROOT looks wrong when run via sudo
APP_DIR="$ROOT"

UNIT=/etc/systemd/system/alpr.service

cat >"$UNIT" <<EOF
[Unit]
Description=alpr API (Linux CPU)
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${APP_DIR}
EnvironmentFile=-${APP_DIR}/.env
Environment=HOST=0.0.0.0
Environment=PORT=8000
Environment=ALPR_MODE=fast
Environment=ALPR_PROVIDERS=cpu
ExecStart=${APP_DIR}/.venv/bin/python -m python.api
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now alpr
systemctl --no-pager status alpr || true

echo ""
echo "Service installed. Check:"
echo "  sudo systemctl status alpr"
echo "  curl http://127.0.0.1:8000/health"
echo "  journalctl -u alpr -f"
