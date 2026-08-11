#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-}"

if command -v systemd-inhibit >/dev/null 2>&1; then
    echo "Mantendo a máquina acordada (systemd-inhibit)..."
    echo "Pressione Ctrl+C para parar."
    if [[ -n "$DURATION" ]]; then
        systemd-inhibit --what=sleep:idle --why="Treinamento de IA em andamento" --mode=block sleep "$DURATION"
    else
        systemd-inhibit --what=sleep:idle --why="Treinamento de IA em andamento" --mode=block sleep infinity
    fi
elif command -v caffeinate >/dev/null 2>&1; then
    echo "Mantendo a máquina acordada (caffeinate)..."
    caffeinate -d
else
    echo "Nenhuma ferramenta encontrada (systemd-inhibit/caffeinate)."
    echo "Desativando sleep do systemd manualmente..."
    sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null
    echo "Sleep desativado. Reative com:"
    echo "  sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target"
fi
