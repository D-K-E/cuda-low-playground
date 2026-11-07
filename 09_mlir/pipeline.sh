#!/usr/bin/env bash
# pipeline.sh: run ops.mlir through each lowering stage and keep every stage in out/.
# Pass names are LLVM 19's. Not tested yet: if a pass is unknown, find its current name
# with `$MLIR_OPT --help | grep <word>` and fix it here.
# Usage: ./pipeline.sh [input.mlir]   (default ops.mlir)
set -euo pipefail
# Tool lookup: $LLVM_BIN if set, else /usr/lib/llvm-19/bin, else whatever is on PATH.
if [ -n "${LLVM_BIN:-}" ]; then BIN=$LLVM_BIN
elif [ -x /usr/lib/llvm-19/bin/mlir-opt ]; then BIN=/usr/lib/llvm-19/bin
else BIN=$(dirname "$(command -v mlir-opt || echo /missing/mlir-opt)"); fi
MLIR_OPT=$BIN/mlir-opt
MLIR_TRANSLATE=$BIN/mlir-translate
[ -x "$MLIR_OPT" ] || { echo "mlir-opt not found: set LLVM_BIN (see colab/setup.sh)" >&2; exit 1; }
IN=${1:-ops.mlir}
mkdir -p out

stage() {  # stage <name> <input file> <passes...>
  local name=$1 src=$2
  shift 2
  "$MLIR_OPT" "$src" "$@" -o "out/$name.mlir"
  echo "out/$name.mlir: $(grep -c 'linalg\.' "out/$name.mlir" || true) linalg ops," \
       "$(grep -c 'memref.alloc' "out/$name.mlir" || true) allocs," \
       "$(grep -c 'scf.for' "out/$name.mlir" || true) loops"
}

cp "$IN" out/00_input.mlir
stage 01_fused       out/00_input.mlir  --linalg-fuse-elementwise-ops --canonicalize --cse
stage 02_bufferized  out/01_fused.mlir  --empty-tensor-to-alloc-tensor \
                                        --one-shot-bufferize="bufferize-function-boundaries" \
                                        --canonicalize --cse
stage 03_loops       out/02_bufferized.mlir --convert-linalg-to-loops --canonicalize
stage 04_llvm        out/03_loops.mlir  --convert-scf-to-cf --expand-strided-metadata \
                                        --lower-affine --convert-arith-to-llvm \
                                        --convert-cf-to-llvm --finalize-memref-to-llvm \
                                        --convert-func-to-llvm --reconcile-unrealized-casts
"$MLIR_TRANSLATE" --mlir-to-llvmir out/04_llvm.mlir -o out/05_llvm.ll
echo "out/05_llvm.ll: LLVM IR"
