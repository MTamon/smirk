#!/usr/bin/env bash
# SMIRK install script for Python 3.11 / PyTorch 2.9.1 / CUDA 12.8.
#
# Preconditions (verified before running this script):
#   - Python 3.11 is active (e.g. `python3.11 -m venv .venv && source .venv/bin/activate`).
#   - System CUDA Toolkit 12.8 is installed (nvcc on PATH, or CUDA_HOME set).
#   - gcc-11 / g++-11 are installed (Ubuntu 22.04: `sudo apt install gcc-11 g++-11`).
#
# This script mirrors MTamon/DECA (cuda128) install_128.sh and
# MTamon/FlashAvatar (release/cuda128-fixed) install_128.sh as closely as
# possible so SMIRK can coexist with DECA / FlashAvatar / FLARE.
# `--no-deps` is used on the DECA-aligned block to prevent pip from mutating
# the pin set via transitive resolution.
#
# After the pinned block, SMIRK-specific packages (timm, mediapipe,
# albumentations, omegaconf, pytorch_lightning, gdown) are installed with
# dependency resolution enabled — they are not part of DECA/FlashAvatar and
# their transitive deps are already satisfied by the pinned block.

set -euo pipefail

# ----------------------------------------------------------------------------
# Toolchain setup (only required if building CUDA extensions; kept for
# symmetry with DECA128 / FlashAvatar128 so the same shell session works for
# all three repos).
# ----------------------------------------------------------------------------
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "[install_128.sh] WARNING: CUDA_HOME is not set and /usr/local/cuda-12.8 was not found."
        echo "[install_128.sh]          Set CUDA_HOME manually to your CUDA 12.8 install path before rerunning."
        exit 1
    fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

# Turing 7.5, Ampere 8.0/8.6, Ada 8.9, Hopper 9.0, Blackwell 12.0 (RTX 5090).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0}"
export FORCE_CUDA=1

echo "[install_128.sh] CC=${CC} CXX=${CXX}"
echo "[install_128.sh] CUDA_HOME=${CUDA_HOME}"
echo "[install_128.sh] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvcc --version || { echo "[install_128.sh] nvcc not found on PATH"; exit 1; }

# ----------------------------------------------------------------------------
# 1. Upgrade pip.
# ----------------------------------------------------------------------------
python -m pip install --upgrade pip==25.2

# ----------------------------------------------------------------------------
# 2. chumpy from GitHub main (numpy 2.x compatible; same source as DECA128).
# ----------------------------------------------------------------------------
pip install git+https://github.com/mattloper/chumpy.git

# ----------------------------------------------------------------------------
# 3. DECA128-aligned pinned dependencies (identical set to
#    MTamon/DECA/cuda128/install_128.sh and FlashAvatar128).
# ----------------------------------------------------------------------------
pip install --no-deps Cython==0.29.35
pip install --no-deps face-alignment==1.4.1
pip install --no-deps filelock==3.20.0
pip install --no-deps fsspec==2025.10.0
pip install --no-deps fvcore==0.1.5.post20221221
pip install --no-deps ImageIO==2.37.2
pip install --no-deps iopath==0.1.10
pip install --no-deps Jinja2==3.1.6
pip install --no-deps joblib==1.5.2
pip install --no-deps kornia==0.8.2
pip install --no-deps kornia_rs==0.1.10
pip install --no-deps lazy_loader==0.4
pip install --no-deps llvmlite==0.45.1
pip install --no-deps MarkupSafe==3.0.3
pip install --no-deps mpmath==1.3.0
pip install --no-deps networkx==3.5
pip install --no-deps ninja==1.13.0
pip install --no-deps numba==0.62.1
pip install --no-deps numpy==2.2.6
pip install --no-deps nvidia-cublas-cu12==12.8.4.1
pip install --no-deps nvidia-cuda-cupti-cu12==12.8.90
pip install --no-deps nvidia-cuda-nvrtc-cu12==12.8.93
pip install --no-deps nvidia-cuda-runtime-cu12==12.8.90
pip install --no-deps nvidia-cudnn-cu12==9.10.2.21
pip install --no-deps nvidia-cufft-cu12==11.3.3.83
pip install --no-deps nvidia-cufile-cu12==1.13.1.3
pip install --no-deps nvidia-curand-cu12==10.3.9.90
pip install --no-deps nvidia-cusolver-cu12==11.7.3.90
pip install --no-deps nvidia-cusparse-cu12==12.5.8.93
pip install --no-deps nvidia-cusparselt-cu12==0.7.1
pip install --no-deps nvidia-nccl-cu12==2.27.5
pip install --no-deps nvidia-nvjitlink-cu12==12.8.93
pip install --no-deps nvidia-nvshmem-cu12==3.3.20
pip install --no-deps nvidia-nvtx-cu12==12.8.90
pip install --no-deps opencv-python==4.12.0.88
pip install --no-deps packaging==25.0
pip install --no-deps pillow==12.0.0
pip install --no-deps portalocker==3.2.0
pip install --no-deps PyYAML==6.0.3
pip install --no-deps scikit-image==0.25.2
pip install --no-deps scikit-learn==1.7.2
pip install --no-deps scipy==1.16.3
pip install --no-deps six==1.17.0
pip install --no-deps sympy==1.14.0
pip install --no-deps tabulate==0.9.0
pip install --no-deps termcolor==3.2.0
pip install --no-deps threadpoolctl==3.6.0
pip install --no-deps tifffile==2025.10.16
pip install --no-deps torch==2.9.1
pip install --no-deps torchvision==0.24.1
pip install --no-deps tqdm==4.67.1
pip install --no-deps triton==3.5.1
pip install --no-deps typing_extensions==4.15.0
pip install --no-deps yacs==0.1.8

# ----------------------------------------------------------------------------
# 4. SMIRK-specific additions (not in DECA128 / FlashAvatar128).
# Installed with normal resolution because their transitive deps are already
# satisfied by the pinned block above.
# ----------------------------------------------------------------------------
# timm 0.9.16 matches the layer naming of the released SMIRK checkpoints.
pip install --no-deps timm==0.9.16
pip install --no-deps huggingface_hub==0.27.1
pip install --no-deps safetensors==0.4.5
# mediapipe 0.10.14 is the minimum version with Python 3.11 + numpy 2.x wheels.
pip install --no-deps absl-py==2.1.0
pip install --no-deps attrs==24.2.0
pip install --no-deps flatbuffers==24.3.25
pip install --no-deps jax==0.4.30
pip install --no-deps jaxlib==0.4.30
pip install --no-deps ml_dtypes==0.4.1
pip install --no-deps opt_einsum==3.4.0
pip install --no-deps protobuf==4.25.5
pip install --no-deps sounddevice==0.5.1
pip install --no-deps sentencepiece==0.2.0
pip install --no-deps mediapipe==0.10.14
# albumentations 1.4.18 is the minimum numpy 2.x compatible version.
pip install --no-deps pydantic==2.9.2
pip install --no-deps pydantic_core==2.23.4
pip install --no-deps annotated-types==0.7.0
pip install --no-deps albucore==0.0.19
pip install --no-deps eval_type_backport==0.2.2
pip install --no-deps stringzilla==3.11.3
pip install --no-deps albumentations==1.4.18
pip install --no-deps omegaconf==2.3.0
pip install --no-deps antlr4-python3-runtime==4.9.3
# pytorch_lightning 2.5.x is for training; inference scripts do not need it
# but it is harmless to install.
pip install --no-deps lightning-utilities==0.11.9
pip install --no-deps torchmetrics==1.6.0
pip install --no-deps pytorch_lightning==2.5.2
pip install --no-deps gdown==5.2.0
pip install --no-deps beautifulsoup4==4.12.3
pip install --no-deps soupsieve==2.6
pip install --no-deps requests==2.32.3
pip install --no-deps charset_normalizer==3.4.0
pip install --no-deps idna==3.10
pip install --no-deps urllib3==2.2.3
pip install --no-deps certifi==2024.8.30

echo "[install_128.sh] done. Quick sanity check:"
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("torch.cuda     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
try:
    import timm
    print("timm           :", timm.__version__)
except Exception as e:
    print("timm           :", repr(e))
try:
    import mediapipe as mp
    print("mediapipe      :", mp.__version__)
except Exception as e:
    print("mediapipe      :", repr(e))
try:
    import numpy as np
    print("numpy          :", np.__version__)
except Exception as e:
    print("numpy          :", repr(e))
try:
    from src.smirk_encoder import SmirkEncoder
    enc = SmirkEncoder()
    print("SmirkEncoder   : ok (params:", sum(p.numel() for p in enc.parameters()), ")")
except Exception as e:
    print("SmirkEncoder   :", repr(e))
PY
