#!/usr/bin/env bash
# Demo-only asset preparation for SMIRK on cuda128.
#
# This script is a slimmer subset of quick_install.sh: it downloads only
# the files required to run demo.py / demo_video.py / demo_save_flame.py
# / demo_webcam.py. Training-only assets (EMOCA ResNet50, MICA,
# expression templates) are NOT downloaded here — use quick_install.sh
# when those are needed.
#
# Downloads:
#   1. FLAME2020 generic_model.pkl            (credential-gated, flame.is.tue.mpg.de)
#        destination: assets/FLAME2020/generic_model.pkl
#        needed by:   demo.py, demo_video.py, demo_webcam.py (mesh render)
#        NOT needed by demo_save_flame.py (pure feature save).
#   2. MediaPipe Face Landmarker task file    (public, storage.googleapis.com)
#        destination: assets/face_landmarker.task
#        needed by:   all four demos (face detection / blendshapes)
#   3. SMIRK pretrained model SMIRK_em1.pt    (gdown, Google Drive)
#        destination: pretrained_models/SMIRK_em1.pt
#        needed by:   all four demos
#
# The FLAME2020 step can be skipped via --no_flame if you only plan to run
# demo_save_flame.py (feature save) or demo_webcam.py --no_render.

set -euo pipefail

WITH_FLAME=1
for arg in "$@"; do
    case "$arg" in
        --no_flame) WITH_FLAME=0 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "[prepare_demos.sh] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

urle () {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

mkdir -p assets pretrained_models

# ----------------------------------------------------------------------------
# 1. FLAME2020 (credential-gated)
# ----------------------------------------------------------------------------
if [ "$WITH_FLAME" = "1" ]; then
    if [ -f assets/FLAME2020/generic_model.pkl ]; then
        echo '[prepare_demos.sh] FLAME2020 already present, skipping.'
    else
        echo
        echo 'FLAME2020 is credential-gated. Register at https://flame.is.tue.mpg.de/'
        read -r -p 'Username (FLAME): ' FLAME_USER
        read -r -s -p 'Password (FLAME): ' FLAME_PASS
        echo
        FLAME_USER_ENC="$(urle "$FLAME_USER")"
        FLAME_PASS_ENC="$(urle "$FLAME_PASS")"

        mkdir -p assets/FLAME2020
        echo '[prepare_demos.sh] Downloading FLAME2020.zip ...'
        wget --post-data "username=${FLAME_USER_ENC}&password=${FLAME_PASS_ENC}" \
             'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
             -O './FLAME2020.zip' --no-check-certificate --continue
        unzip -o FLAME2020.zip -d assets/FLAME2020/
        rm FLAME2020.zip
    fi
else
    echo '[prepare_demos.sh] --no_flame given; skipping FLAME2020.'
fi

# ----------------------------------------------------------------------------
# 2. MediaPipe Face Landmarker task file (public)
# ----------------------------------------------------------------------------
if [ -f assets/face_landmarker.task ]; then
    echo '[prepare_demos.sh] face_landmarker.task already present, skipping.'
else
    echo '[prepare_demos.sh] Downloading face_landmarker.task ...'
    wget -q --show-progress \
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' \
        --directory-prefix assets/
fi

# ----------------------------------------------------------------------------
# 3. SMIRK pretrained checkpoint (gdown)
# ----------------------------------------------------------------------------
if [ -f pretrained_models/SMIRK_em1.pt ]; then
    echo '[prepare_demos.sh] SMIRK_em1.pt already present, skipping.'
else
    echo '[prepare_demos.sh] Downloading SMIRK_em1.pt via gdown ...'
    # --id is deprecated in gdown 5.x; use the explicit URL form instead.
    gdown --fuzzy 'https://drive.google.com/uc?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE' \
          -O pretrained_models/SMIRK_em1.pt
fi

echo
echo '[prepare_demos.sh] Done. Asset summary:'
ls -lh assets/face_landmarker.task pretrained_models/SMIRK_em1.pt 2>/dev/null || true
if [ "$WITH_FLAME" = "1" ]; then
    ls -lh assets/FLAME2020/generic_model.pkl 2>/dev/null || \
        echo '  WARN: assets/FLAME2020/generic_model.pkl not found.'
fi

cat <<'USAGE'

Next steps — run any of the demos via the shell-wrapper launchers
(these scope the NVIDIA EGL vendor env vars to the subshell so the
MediaPipe GPU delegate actually opens a GL context on the RTX and
the variables vanish from the user's shell afterwards):

  # Single image (mesh overlay)
  bash demos/run_demo.sh --input_path samples/test_image1.png --crop

  # Video file (mesh overlay, side-by-side mp4)
  bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop

  # Video file -> save FLAME params only (no render), with benchmark
  bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
      --crop --benchmark

  # Video file -> save FLAME params + eyes_pose/eyelids from blendshapes
  bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
      --crop --with_eye_pose --benchmark --mp_delegate gpu

  # Real-time webcam with live mesh overlay (MediaPipe GPU delegate)
  bash demos/run_demo_webcam.sh --mp_delegate gpu

  # Real-time webcam, CPU-only benchmark mode (no mesh render)
  bash demos/run_demo_webcam.sh --device cpu --no_render --mp_delegate cpu

See demos/demos.md for a detailed guide.

USAGE
