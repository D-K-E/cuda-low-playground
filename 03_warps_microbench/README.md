# 03 Warps, cooperative groups and microbenchmarks

**Goal:** measure, on your own hardware, the latencies and throughputs that the job
description calls "intuition": kernel launch vs CUDA graph launch, warp-level and
block-level synchronisation, shared-memory and global-memory latency, and (on the cloud
GPU) asynchronous copies. You'll quote these numbers in interviews.

**CV line:** "Wrote CUDA microbenchmarks for launch overhead, CUDA graphs, warp shuffles,
barriers and memory latency on Maxwell and Ampere; used them to guide kernel design."

**Read:** *CUDA C++ Programming Guide*, the sections on warp shuffle functions,
cooperative groups and CUDA graphs. Optional: Jia et al., *Dissecting the NVIDIA Volta GPU
Architecture via Microbenchmarking*, for the pointer-chasing method.

## How to measure something that takes nanoseconds

- **Latency** (time for one dependent operation): build a chain where each step needs the
  previous result, e.g. `x = __shfl_sync(~0u, x, (lane + 1) & 31)` 1000 times in a row. Time
  it with `clock64()` inside the kernel, divide by the count, report cycles.
- **Throughput** (independent operations per cycle): many independent chains per thread and
  many warps, then total ops / total time.
- **Keep the compiler honest:** write the result to global memory, or the loop is deleted.
  Check the SASS (module 02) to confirm the instructions are really there.

## Boilerplate

- `launch.cu`: experiments 1-2 (stream launches, launch + sync, graph instantiate and
  launch). The kernels-per-graph sweep is a TODO.
- `chase.cu`: experiment 3, pointer chasing from 4 KB to `max_mb`. `make chase-l1` builds
  it with `-Xptxas -dlcm=ca`: Maxwell caches global loads in L2 only unless told
  otherwise, so compare the two curves for the L1 step.
- Experiments 4-9: copy the pattern from `chase.cu` (one thread, `clock64()`, a
  dependent chain, write the result out).

## Experiments

| # | measure | method | where |
|---|---|---|---|
| 1 | empty kernel launch | 10k launches of an empty kernel, host-side timer, per-launch µs | local |
| 2 | CUDA graph launch | capture the same 10k kernels with stream capture, instantiate once, time `cudaGraphLaunch`; plot launch cost vs kernels per graph | local |
| 3 | global memory latency | pointer chasing over arrays from 4 KB to 1 GB: the L1, L2 and DRAM steps appear in the curve | local |
| 4 | shared memory latency | pointer chasing in `__shared__` | local |
| 5 | `__shfl_sync` latency and throughput | dependent chain, then independent chains | local |
| 6 | `__syncthreads` cost | blocks of 64 to 1024 threads, barrier in a loop | local |
| 7 | cooperative groups | `thread_block_tile<32>` reduce vs hand-written `__shfl_down_sync`: same SASS? `this_grid().sync()` needs `sm_60`+: try it locally and record the error | local, then cloud |
| 8 | atomics | same-address vs different-address `atomicAdd`, in global and shared memory | local |
| 9 | `cp.async` | global to shared copy, synchronous vs `cp.async` with 2 and 4 stages, hiding a fixed compute loop | cloud |

## Questions to answer from your numbers

- How many tiny kernels per second can a single stream launch? At what kernel duration
  does launch overhead stop mattering? What does a graph save per kernel?
- How many cycles of DRAM latency does a warp need to hide, and how many warps per SM does
  that take (Little's law: latency x bandwidth = bytes in flight)?
- Why does Volta+ need the mask in `__shfl_sync`, and why did Maxwell code without it work?

## Deliverable: `RESULTS.md`

One table of all measurements (value, unit, GPU), the latency-vs-array-size chart, the
launch-cost chart, and the answers to the three questions.
