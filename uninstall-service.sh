#!/usr/bin/env bash
# Linux (systemd) uninstaller — counterpart to uninstall-service.ps1. Stops and
# removes the service unit. Leaves the venv, the HuggingFace model cache, and
# the runtime data (DBs, captures, logs) in place — same policy as the .ps1.
set -euo pipefail

SERVICE_NAME="whisper-api"

if [ "$(id -u)" -ne 0 ]; then
  echo "Elevating with sudo..."
  exec sudo -E "$0" "$@"
fi

UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

if systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
  echo "Stopping + disabling ${SERVICE_NAME} ..."
  systemctl disable --now "${SERVICE_NAME}" || true
else
  echo "${SERVICE_NAME} is not installed; nothing to stop."
fi

if [ -f "$UNIT" ]; then
  echo "Removing $UNIT ..."
  rm -f "$UNIT"
  systemctl daemon-reload
fi

echo "Done. (venv, model cache, and *.local.sqlite3 data were left intact.)"
