# 06 Debugging and profiling tickets

**Goal:** fix planted bugs and performance problems with the right tool each time, and
write each one up like an incident: symptom, tool, root cause, fix, regression check.

**CV line:** "Diagnosed CUDA race conditions, out-of-bounds accesses and performance
regressions with compute-sanitizer, CUDA-GDB, Nsight Systems and Nsight Compute."

**Read:** the compute-sanitizer docs (memcheck, racecheck, initcheck, synccheck), the
CUDA-GDB docs (switching focus to a block, thread or warp), the Nsight Systems user guide
(the CUDA trace and NVTX ranges).

## Method (the same one as `cpp/debug/`)

Reproduce first. Pick the cheapest tool that answers the question. State the root cause
with evidence. Keep the fix minimal, and add a check that fails without it.

## You already have most of the tool drills

- `src/mlprep/cuda/item-debugging/ex01-ex10`: memcheck, racecheck, initcheck, synccheck,
  sticky errors, cuda-gdb (method in `cuda-prep_v2.md` section D).
- `src/mlprep/cpp/debug/cuda/C01-C04`: the same bugs as tickets in a trading-desk context,
  answers in `cpp/debug/solutions/`.

For the public repo, port three or four of those here with write-ups, and add the ones
they don't cover: **T5** (Nsight Systems timeline) and **T7** (register spills).

## Tickets

Each ticket: a small `.cu` file with one planted bug and a symptom description. Write your
own by breaking module 01 and 03 code on purpose, or ask for a generated set.

| # | symptom | tool to reach for | planted cause |
|---|---|---|---|
| T1 | sum is wrong only for `n` not a multiple of 256 | compute-sanitizer memcheck | missing bounds check |
| T2 | sum is wrong about 1 run in 20 | racecheck | missing `__syncthreads` in the tree |
| T3 | kernel hangs | CUDA-GDB: which threads wait where? | `__syncthreads` inside a divergent branch |
| T4 | garbage in the first output only | initcheck | shared memory read before it's written |
| T5 | correct, but the pipeline is 3x slower than the kernels add up to | Nsight Systems timeline | a `cudaMemcpy` (synchronous) between kernels; pageable host memory |
| T6 | a new version of transpose is 2x slower | `nvprof` locally, Nsight Compute on the cloud GPU | bank conflicts reintroduced (no padding) |
| T7 | slower than expected after adding a feature | `-Xptxas -v` and the profiler | register spills: occupancy dropped |

Use NVTX ranges (`nvtxRangePush` / `nvtxRangePop`) in T5 so the timeline shows your phases
by name.

## Deliverable: `RESULTS.md`

One short write-up per ticket: symptom, tool and the exact output that pointed at the
cause, root cause, fix, how you checked it. Screenshots of the T5 timeline before and after.
