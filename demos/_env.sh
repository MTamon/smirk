#!/usr/bin/env bash
# demos/_env.sh — per-launch environment for SMIRK demo shell wrappers.
#
# This file is `source`d by run_demo*.sh. Every variable it exports lives
# only inside the wrapper's subshell — when the wrapper exits, the user's
# interactive shell is untouched. That scoping is the whole point of
# launching the demos through shell scripts instead of invoking Python
# directly: we enable NVIDIA-specific EGL/GLX vendor routing for the
# duration of the run and nothing more.
#
# --- Why these variables matter ---
#
# MediaPipe's FaceLandmarker (and any TFLite model compiled with the GPU
# delegate) creates an **EGL** context at detector-construction time. On
# Linux, EGL/GLX go through libglvnd, which dispatches to a vendor driver
# based on JSON configs in /usr/share/glvnd/egl_vendor.d/ — the default
# ordering lets MESA try first. On a pure-NVIDIA host without MESA DRI
# drivers (common on cloud and ML rigs), libglvnd tries MESA, fails with
#
#    libEGL warning: MESA-LOADER: failed to open radeonsi / swrast
#    GPU support is not available: Unable to initialize EGL
#
# MediaPipe then silently **falls back to the CPU XNNPACK delegate**,
# which drops demo_webcam.py from ~100 FPS to ~20 FPS without any error.
#
# Forcing libglvnd to pick the NVIDIA vendor JSON skips the MESA probe
# entirely and lets MediaPipe open a real GL context on the NVIDIA GPU.
# No system-wide config change is needed — just these two env vars for
# the lifetime of the wrapper.
#
# --- Scope to remember for downstream integration ---
#
# The same pattern applies to FLARE / DECA / FlashAvatar integrations
# that embed MediaPipe Tasks: whoever launches the Python process must
# export __EGL_VENDOR_LIBRARY_FILENAMES before the first
# ``FaceLandmarker`` is created. Setting it later has no effect because
# libglvnd caches the vendor dispatch on first use.

_NVIDIA_EGL_JSON_CANDIDATES=(
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    "/etc/glvnd/egl_vendor.d/10_nvidia.json"
    "/usr/local/share/glvnd/egl_vendor.d/10_nvidia.json"
)

_nv_json=""
for _c in "${_NVIDIA_EGL_JSON_CANDIDATES[@]}"; do
    if [ -f "$_c" ]; then
        _nv_json="$_c"
        break
    fi
done

if [ -n "$_nv_json" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$_nv_json"
    export __GLX_VENDOR_LIBRARY_NAME="nvidia"
    echo "[demos/_env.sh] EGL vendor pinned to NVIDIA: $_nv_json"
else
    echo "[demos/_env.sh] WARN: NVIDIA EGL vendor JSON not found in any of:" >&2
    for _c in "${_NVIDIA_EGL_JSON_CANDIDATES[@]}"; do
        echo "                - $_c" >&2
    done
    echo "[demos/_env.sh] WARN: MediaPipe GPU delegate will likely fall back to CPU." >&2
fi

# Keep Python bytecode + CUDA allocator behaviour predictable for
# short-lived demo runs. Harmless if unset, no pollution after exit.
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# Ensure `python demos/demo_*.py` and `from src.*` work regardless of the
# caller's PYTHONPATH. The sys.path bootstrap inside each demo script
# also handles this, but exporting the variable is a cheap belt-and-braces
# for downstream tooling that imports these modules.
_HERE_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT_ABS="$(cd "$_HERE_ABS/.." && pwd)"
export PYTHONPATH="${_ROOT_ABS}${PYTHONPATH:+:$PYTHONPATH}"

unset _c _nv_json _NVIDIA_EGL_JSON_CANDIDATES _HERE_ABS _ROOT_ABS
