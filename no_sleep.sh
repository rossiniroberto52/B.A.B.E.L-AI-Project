#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-}"
PID_FILE="/tmp/no_sleep_$$.pid"

if command -v systemd-inhibit >/dev/null 2>&1; then
    if [[ -n "$DURATION" ]]; then
        echo "Mantendo a máquina acordada por $DURATION (background)..."
        systemd-inhibit --what=sleep:idle --why="Treinamento de IA em andamento" --mode=block sleep "$DURATION" &
    else
        echo "Mantendo a máquina acordada (background)..."
        systemd-inhibit --what=sleep:idle --why="Treinamento de IA em andamento" --mode=block sleep infinity &
    fi
    echo $! > "$PID_FILE"
    echo "PID: $(cat $PID_FILE) - Para parar: kill $(cat $PID_FILE)"
elif command -v caffeinate >/dev/null 2>&1; then
    echo "Mantendo a máquina acordada (caffeinate, background)..."
    caffeinate -d &
    echo $! > "$PID_FILE"
    echo "PID: $(cat $PID_FILE) - Para parar: kill $(cat $PID_FILE)"
else
    echo "Nenhuma ferramenta encontrada (systemd-inhibit/caffeinate)."
    echo "Desativando sleep do systemd manualmente..."
    sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null
    echo "Sleep desativado. Reative com:"
    echo "  sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target"
fi
