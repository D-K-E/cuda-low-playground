# gpu-lab: low-level GPU work, measured

A guided project that turns each line of a GPU-systems job description into something
measured and explained. Each module ends in a deliverable (a table, a chart, an annotated
listing) and a CV line that the deliverable backs up.

Planned to become its own public repo; each module folder is self-contained so it can be
moved out one at a time.

## Hardware

Two machines, chosen per step:

| where | GPU | use it for |
|---|---|---|
| **local** | Quadro M-series (Maxwell; `cuda-prep_v2.md` records an M2000M, `sm_50`) | memory hierarchy, warps, PTX/SASS, CUDA graphs, CUB/Thrust/cuBLAS, debugging |
| **cloud, 1 GPU** (Ampere `sm_80`/`sm_86` or Ada `sm_89`; Hopper `sm_90` optional) | rented by the hour | Tensor Cores, `cp.async`, Nsight Compute, Triton, CUTLASS 3 |
| **cloud, multi-GPU** (4-8 GPUs with NVLink) | rented for a few hours, scripts ready beforehand | NCCL, `nccl-tests`, NVLink vs PCIe |
| **any CPU machine** | none | MPI collectives, cost models, MLIR |

What Maxwell can't do, and why that matters for the plan (confirm each in module 00):

| feature | needs | on `sm_50` |
|---|---|---|
| Tensor Cores (WMMA, `mma.sync`) | Volta `sm_70`+ | no: cloud |
| `cp.async` asynchronous copies | Ampere `sm_80`+ | no: cloud |
| TMA, `wgmma` | Hopper `sm_90` | no: cloud, optional |
| grid-wide `this_grid().sync()` (cooperative launch) | Pascal `sm_60`+ | no; tile and block groups do work |
| independent thread scheduling (why `__shfl_sync` takes a mask) | Volta `sm_70`+ | no: warps run in lockstep, a useful contrast |
| Nsight Compute | Volta+ | no: use `nvprof` locally, Nsight Compute on the cloud GPU |
| Triton | `sm_75`/`sm_80`+ | no: cloud |
| CUDA toolkit | CUDA 13 dropped Maxwell | **CUDA 12.x**, or **11.8** if `nvprof` is missing from 12.x |

Running one kernel on both `sm_50` and a modern GPU, and explaining the difference, is a
deliverable in its own right (modules 02 and 04).

## Modules

| # | folder | job-description line | deliverable | where |
|---|---|---|---|---|
| 00 | [`00_setup`](00_setup/README.md) | | tool support table for both machines | both |
| 01 | [`01_memory`](01_memory/README.md) | memory hierarchy, warps | reduction and transpose ladders: % of peak bandwidth per step | local |
| 02 | [`02_ptx_sass`](02_ptx_sass/README.md) | PTX, SASS | annotated SASS diff explaining a speedup; `sm_50` vs `sm_80` | local, then cloud |
| 03 | [`03_warps_microbench`](03_warps_microbench/README.md) | warps, cooperative groups, latency of launches, warp sync, async loads | a table of your own measured latencies and throughputs | local, then cloud |
| 04 | [`04_gemm_ladder`](04_gemm_ladder/README.md) | Tensor Cores, async loads, CUTLASS | GEMM at % of cuBLAS, SIMT then Tensor Core | local, then cloud |
| 05 | [`05_libraries`](05_libraries/README.md) | Triton, CUTLASS, CUB, Thrust, cuDNN, cuBLAS | "which library when", with timings | local, then cloud |
| 06 | [`06_debugging`](06_debugging/README.md) | CUDA-GDB, Nsight Systems, Nsight Compute | planted-bug tickets with write-ups | local |
| 07 | [`07_collectives`](07_collectives/README.md) | NCCL / MPI collective algorithms | ring and tree all-reduce + cost model checked against measurements | CPU, then multi-GPU |
| 08 | [`08_cluster_networking`](08_cluster_networking/README.md) | InfiniBand, RoCE, GPUDirect, PXN, rails, NVLink | design notes, topology diagrams, `nccl-tests` sweeps | notes, then multi-GPU |
| 09 | [`09_mlir`](09_mlir/README.md) | ML compilers (MLIR basics) | one layer through fusion, tiling, bufferization and lowering to LLVM IR and PTX, with a per-stage table | CPU, then local GPU |

Suggested order: 00, 01, 02, 03 (local part), 06, 05 (local part), 04 (SIMT part), then
one cloud session for 02/03/04/05's modern-GPU parts, then 07 and 08. Module 09 can go
anywhere after 05, since its fusion accounting builds on fused softmax.

## Every module has the same shape

1. **Goal**, and the CV line it earns.
2. **Read**: one or two sources.
3. **Guided steps**: questions to answer before writing code, then the code.
4. **Boilerplate**, only where setup is the hard part.
5. **Deliverable**: `RESULTS.md` in the module folder, with the numbers and why they came
   out that way.

**Rule for CV lines:** claim only what a `RESULTS.md` shows. "Benchmarked NCCL ring vs tree
on 8x A100 over NVLink" is checkable; "experience with InfiniBand" is not, unless you ran it.

## Build

Each module with code has a `Makefile`. `ARCH` defaults to `sm_50`, which also runs on
`sm_52` cards (same major version), so it fits any Maxwell GPU:

```bash
make                 # local Maxwell
make ARCH=sm_80      # cloud
```

## Basic Terms

- Grid: batch size of work, total number of blocks.
- Block: a batch of work, contains threads that can access to `__shared__`
  memory.
- Warp: a group of 32 threads that runs the same instruction (kernel). Notice
  that this is different than block. Thinking in warps are important, because `if else`
  branches can diverge them. Divergence means part of the threads in the warp
  has to wait while the others do operations in one of the branches.
- Lane: The identifier of a thread in warp. Notice that this is independent of
  `threadIdx` which identifies the thread within a block. Lane is thread
  within a warp.
- L1 cache: has the same hardware space as `__shared__`. The main difference
  is `__shared__` is controlled by the programmer, L1 cache is not. It is
  controlled by the gpu hardware.
- L2 cache: controlled by the gpu hardware.
- external shared memory: `extern __shared__ float s[]` memory, whose size is
  specified during the kernel execution.
- shared memory: `__shared__ float s[32]` memory, faster than global VRAM,
  accessible by all threads in block.
- register: size of a single register is 32-bit/4 byte, meaning 1 `float32` or 1
  `int32`
- thread registers: a group of registers accessible by a single thread. In
  most modern gpu architectures, the number of registers accessible by a
  single thread is 255. So, total of `255 * 4 = 1020` bytes.
- 

### Tips and Tricks

#### Warp Reduce

Most reduction operations benefit from warp usage. General shape of the
algorithm usually looks like:

1. grid stride to fill the registers of a thread block
2. for each lane in warp, access its register and do the reduction operation
3. if you shuffle up, the operation result would be on the last lane of the
warp, if you shuffle down, the operation result would be on the first lane.
Why do this at the warp level ? Because cuda api gives you access to warp
level primitives.
4. Now that a particular lane contains result for each warp, store the warp
result to a shared memory. Size of the shared memory should be at least the number
of warps in a thread block. Usually using the value 32 is nice, because it
allows you to iterate over the shared memory while you are iterating over warp
lanes.
5. When shared memory store is ready, sync threads. Before pursuing further,
we need to make sure that each thread in the block has run. If we don't do
this we won't know if shared memory is filled or not.
6. It is time we read from shared memory at the first warp. The strategy is
very similar to grid stride, we simply do a warp stride where stride is the
warp size (32) and limit is number of warps and the counter starts of the
lane.
7. Then do another reduce op at the warp level and write it per block at first
lane.

An example is `01_memory/reduce.cu:reduce_v8b`
