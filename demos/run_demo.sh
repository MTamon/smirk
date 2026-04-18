#!/usr/bin/env bash
# Launcher for demos/demo.py (single image → FLAME mesh overlay).
# All command-line arguments are passed through to Python.
#
# Env vars set by demos/_env.sh are scoped to this shell script only —
# once it returns, the caller's environment is unchanged.
#
# Example:
#   bash demos/run_demo.sh --input_path samples/test_image1.png --crop

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo.py" "$@"
