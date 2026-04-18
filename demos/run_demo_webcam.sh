#!/usr/bin/env bash
# Launcher for demos/demo_webcam.py
# (real-time webcam or ideal-source video with live FLAME mesh overlay).
#
# **GPU MediaPipe here is almost mandatory**: on a live 1280x720 feed
# at 30 FPS, CPU XNNPACK detection budgets ~15 ms/frame which leaves
# very little for SMIRK + render. The wrappers in demos/_env.sh route
# EGL to the NVIDIA vendor so MediaPipe's GPU delegate actually opens
# a GL context on the RTX instead of falling back to CPU.
#
# Examples:
#   bash demos/run_demo_webcam.sh
#   bash demos/run_demo_webcam.sh --mp_delegate gpu
#   bash demos/run_demo_webcam.sh --source samples/dafoe.mp4 --no_render

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# shellcheck disable=SC1091
. "$HERE/_env.sh"

cd "$ROOT"
exec python "$HERE/demo_webcam.py" "$@"
