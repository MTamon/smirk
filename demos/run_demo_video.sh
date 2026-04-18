#!/usr/bin/env bash
# Launcher for demos/demo_video.py (video → side-by-side FLAME mesh mp4).
# All command-line arguments are passed through to Python.
#
# Example:
#   bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_video.py" "$@"
