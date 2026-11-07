# 02 PTX and SASS: reading what the compiler did

**Goal:** read the PTX and SASS of your own module 01 kernels well enough to explain a
speedup from the instructions, and to say what changes between `sm_50` and `sm_80`.

**CV line:** "Analysed PTX/SASS of hand-written CUDA kernels to explain performance
differences (memory instruction widths, shuffles, unrolling) across Maxwell and Ampere."

**Read:** *PTX ISA* reference, the sections on state spaces and `ld`/`st`/`shfl`.
*CUDA Binary Utilities* (the `cuobjdump` and `nvdisasm` docs) for the SASS opcode list.

## Concepts first

- **PTX** is a virtual ISA: what `nvcc` emits, stable across GPUs. **SASS** is the real
  machine code for one architecture, produced by `ptxas`. Performance lives in SASS.
- Questions: why can a binary built with `-arch=sm_50` still run on an Ampere GPU (hint:
  embedded PTX + JIT)? What does `-gencode arch=compute_50,code=sm_50` embed vs
  `code=compute_50`?

## Steps

1. Dump both levels for your reduction:
   ```bash
   nvcc -O3 -arch=sm_50 -ptx reduce.cu -o reduce.ptx
   nvcc -O3 -arch=sm_50 -cubin reduce.cu -o reduce.cubin
   cuobjdump -sass reduce.cubin > reduce.sass
   nvcc -O3 -arch=sm_50 -Xptxas -v reduce.cu    # registers, shared memory, spills
   ```
2. In `reduce.sass`, find for v3 and v5: the global loads (`LDG`), shared loads/stores
   (`LDS`/`STS`), barriers (`BAR.SYNC`), shuffles (`SHFL`), and the loop. Count
   instructions per element in the main loop.
3. Make two controlled changes and diff the SASS:
   - load `float4` instead of `float` in v6: do you see `LDG.E.128` instead of `LDG.E`?
   - add `#pragma unroll 4` to the grid-stride loop: how many `LDG` are in flight now?
   Measure each and connect the SASS change to the time change.
4. Optional: one line of inline PTX, e.g. read `%laneid` or `%smid` with `asm volatile`,
   and show that it matches `threadIdx.x % 32`.
5. Cloud: build the same file with `-arch=sm_80` and compare the SASS of v5 and v6.
   Questions: what's different in the shuffles? Are there new load instructions?

## Deliverable: `RESULTS.md`

- An annotated SASS excerpt for one kernel's main loop: each line with a comment.
- A before/after table for the `float4` and unroll changes: SASS difference, time.
- A short `sm_50` vs `sm_80` comparison.
