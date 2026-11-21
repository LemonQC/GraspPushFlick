#!/usr/bin/env bash
# Source this file. HOME and desktop authentication remain untouched.
export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/cache/conda"
export CONDA_ENVS_PATH="$PROJECT_ROOT/.conda"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/extensions"
export TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg/cache"
export XDG_DATA_HOME="$PROJECT_ROOT/cache/xdg/data"
export XDG_CONFIG_HOME="$PROJECT_ROOT/cache/xdg/config"
export CUDA_CACHE_PATH="$PROJECT_ROOT/cache/nvidia/ComputeCache"
export NVIDIA_CACHE_DIR="$PROJECT_ROOT/cache/nvidia"
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_PATH="$PROJECT_ROOT/cache/nvidia/GLCache"
export OPTIX_CACHE_PATH="$PROJECT_ROOT/cache/nvidia/OptixCache"
export TMPDIR="$PROJECT_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/cache/pycache"
export APFG_PYTHON="$PROJECT_ROOT/.conda/isaac-apfg/bin/python"
export ISAAC_PATH="$PROJECT_ROOT/third_party/isaacsim"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$HF_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$OPTIX_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR" \
  "$PROJECT_ROOT/.runtime_home" "$PROJECT_ROOT/outputs/environment_check"
