#!/usr/bin/env bash
# Colab/Kaggle setup: show the GPU, install MLIR tools, print the ARCH that make will use.
# Usage (notebook cell):  !bash gpu-lab/colab/setup.sh
set -uo pipefail
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || echo "no GPU visible (CPU runtime: only MLIR/MPI modules will work)"
nvcc --version | tail -n2 || echo "nvcc not found"

# MLIR: Ubuntu's own mlir-*-tools first, then apt.llvm.org as a fallback.
if ! command -v mlir-opt >/dev/null && ! ls /usr/lib/llvm-*/bin/mlir-opt >/dev/null 2>&1; then
  sudo_="" ; [ "$(id -u)" -ne 0 ] && sudo_="sudo"
  $sudo_ apt-get update -qq
  for v in 19 18 17 20; do
    if $sudo_ apt-get install -y -qq "mlir-$v-tools" >/dev/null 2>&1; then echo "installed mlir-$v-tools"; break; fi
  done
fi
mlir=$(ls /usr/lib/llvm-*/bin/mlir-opt 2>/dev/null | sort -V | tail -n1)
[ -n "${mlir:-}" ] && echo "export LLVM_BIN=$(dirname "$mlir")" > /tmp/llvm_bin.sh && echo "MLIR at $mlir (source /tmp/llvm_bin.sh)" \
  || echo "mlir-opt not installed; see 09_mlir/README.md"
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r')
echo "make will use ARCH=sm_${arch:-50}  (override: make ARCH=sm_XX)"
