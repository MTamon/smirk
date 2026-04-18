#!/usr/bin/env bash
# Launcher for demos/demo_save_flame.py
# (video → FLAME parameter .pt/.npz save, no render).
#
# Example:
#   bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
#       --crop --benchmark --mp_delegate gpu

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_save_flame.py" "$@"
