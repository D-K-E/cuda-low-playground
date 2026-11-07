# 09 MLIR: fusion and lowering, the basics

**Goal:** take one small layer (matmul + bias + ReLU) through MLIR by hand: fuse the
elementwise ops, tile and fuse the epilogue into the matmul, bufferize, lower to loops and
LLVM IR, and (on the GPU machine) lower to PTX. At each stage, say what changed and why
it matters for performance. This is the machinery under Triton, IREE, torch-mlir and
XLA, at the level most ML-compiler conversations start.

**CV line:** "Lowered a matmul + bias + ReLU layer through MLIR (linalg on tensors to
loops to LLVM IR, and to NVVM/PTX), applying elementwise fusion, tiling with
producer-consumer fusion via the transform dialect, and bufferization."

**Read:** the MLIR *Toy tutorial*, chapters 1-6 (dialects, pattern rewrites, lowering). The
*Linalg dialect rationale* (why structured ops make fusion and tiling easy). The
*Bufferization* doc, the one-shot bufferize overview.

**Tools:** LLVM 19's MLIR (`sudo apt install mlir-19-tools`, then use
`/usr/lib/llvm-19/bin/mlir-opt`). Pass names change between LLVM versions: when a pass
isn't found, run `mlir-opt --help | grep <word>`. `pipeline.sh` defaults to the LLVM 19 names.
**`ops.mlir` and `pipeline.sh` have not been run yet:** getting them to run is the first
step, and fixing a pass name is part of it.

## Concepts first, as questions to answer in `RESULTS.md`

1. **Dialects and progressive lowering:** `linalg` (structured ops), `tensor` (values),
   `memref` (buffers), `scf` (loops), `arith`, `gpu`, `nvvm`, `llvm`. Why lower in several
   small steps instead of one big one? What can you still do at `linalg` level that you
   can't at loop level?
2. **`linalg.generic`:** what do the indexing maps and iterator types (`parallel`,
   `reduction`) in `ops.mlir` say? Write the bias add as a plain loop nest by hand.
3. **Tensors vs buffers:** why do fusion and tiling happen on tensors, before
   bufferization? What does bufferization have to decide (in-place or a copy)?
4. **Fusion:** in `ops.mlir`, how many times does the unfused version read and write a
   64 x 256 intermediate? After elementwise fusion? After fusing the epilogue into the
   matmul tiles? This is the same accounting as fused softmax in module 05.

## Steps

1. **Read the input.** `ops.mlir` has `@mlp`: `linalg.fill`, `linalg.matmul`, then two
   `linalg.generic`s (bias add, ReLU). Answer questions 1 and 2.
2. **Elementwise fusion.**
   `./pipeline.sh` writes every stage to `out/`. Diff `out/00_input.mlir` with
   `out/01_fused.mlir`: the two generics become one. Why isn't the matmul fused too? (It
   isn't elementwise: each output needs a whole row and column.)
3. **Generalise.** `mlir-opt --linalg-generalize-named-ops ops.mlir`: `linalg.matmul` is
   just a `linalg.generic` with a `reduction` iterator. Read its indexing maps.
4. **Tile and fuse the epilogue** with the transform dialect. Write `tile_fuse.mlir`
   yourself: a `transform.named_sequence @__transform_main` that matches the last
   `linalg.generic` (the fused bias + ReLU from step 2), tiles it into 16 x 32 tiles, and
   fuses its producers (matmul, fill) into the tile loop. Look up these ops in your LLVM
   version's Transform dialect docs: `transform.structured.match`,
   `transform.structured.tile_using_forall` (or `tile_using_for`) and
   `transform.structured.fuse_into_containing_op`. Apply it with
   `mlir-opt out/01_fused.mlir --transform-preload-library="transform-library-paths=tile_fuse.mlir" --transform-interpreter --canonicalize --cse`.
   Afterwards, each tile computes its piece of the matmul and applies bias and ReLU while
   the data is still in registers or cache. Count loops and intermediate tensors before
   and after.
5. **Bufferize and lower** (the rest of `pipeline.sh`): one-shot bufferization (look for
   `memref.alloc`: how many temporaries are left?), then loops (`scf.for`), then the
   LLVM dialect, then LLVM IR with `mlir-translate`. Compare the alloc count for the
   unfused version (copy the script and skip stage 01), the fused one, and the tiled one
   (`./pipeline.sh` on the output of step 4).
6. **Run it (optional, CPU).** Add a `@main` that fills inputs, calls `@mlp` and prints
   with `printMemrefF32`, then run with `mlir-cpu-runner` (renamed `mlir-runner` in LLVM 20)
   and `-shared-libs=/usr/lib/llvm-19/lib/libmlir_runner_utils.so,/usr/lib/llvm-19/lib/libmlir_c_runner_utils.so`.
   Time the unfused and fused builds at a larger size (e.g. 1024 x 1024).
7. **GPU (on the GPU machine).** From the fused, bufferized IR:
   `--convert-linalg-to-parallel-loops`, `--gpu-map-parallel-loops`,
   `--convert-parallel-loops-to-gpu`, `--gpu-kernel-outlining`, then
   `--convert-gpu-to-nvvm` and `--gpu-module-to-binary` with an NVVM target for
   `chip=sm_50`. Find the PTX and compare it with your hand-written kernels (module 02).
8. **Connect to Triton** (cloud, with module 05): Triton is built on MLIR. Dump a Triton
   kernel's stages (`kernel.asm["ttir"]`, `["ttgir"]`, `["llir"]`, `["ptx"]`) and say which
   stage does what you did by hand here.

## Deliverable: `RESULTS.md`

- Answers to the four questions.
- One table, one row per stage (input, fused, tiled + fused, bufferized, loops, LLVM): number
  of ops, loops, `memref.alloc`s, and bytes of intermediate data written.
- Optional: the unfused vs fused run time at 1024 x 1024.
- Three or four annotated excerpts: the fused generic, one tile of the tiled loop, a
  bufferized alloc, and a piece of the PTX.
