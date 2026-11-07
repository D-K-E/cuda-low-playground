# 01 Memory hierarchy: reduction and transpose ladders

**Goal:** take two memory-bound kernels from naive to near peak DRAM bandwidth, one change
at a time, and explain each step with a measured number.

**CV line:** "Optimised CUDA reduction and transpose kernels from X% to Y% of peak DRAM
bandwidth on Maxwell; documented the effect of coalescing, shared-memory bank conflicts,
occupancy and warp shuffles."

**Read:** Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA slides; written
for an older GPU than yours, which helps). *CUDA C++ Best Practices Guide*, the sections on
coalesced access and shared memory.

**Already started?** `src/mlprep/cuda/cuda-prep_v2.md` O1 (reduction) and O4 (transpose),
with scaffolds in `cuda/item-optimization-ladders/`, are the same ladders. Bring any code you
wrote there over here; what this module adds is the `RESULTS.md` write-up.

## Before any code

1. Every element of the input is read once, and the output is one number. So the kernel
   can't run faster than `n * 4 bytes / peak bandwidth`. With `n = 2^24` and your
   `devinfo` peak, what's the time floor in milliseconds?
2. A warp of 32 threads reads 32 consecutive floats. How many bytes is that, and how many
   32-byte memory transactions does it take? Now the same warp reads with a stride of 32
   floats: how many transactions?
3. Shared memory on Maxwell has 32 banks, each 4 bytes wide. Threads `tid` access
   `smem[2 * s * tid]`. For `s = 1`, which threads in a warp hit the same bank?

Write your answers in `RESULTS.md` before measuring, then compare.

## Part A: reduction (`reduce.cu`)

`reduce.cu` has the harness (timing, correctness check, GB/s and % of peak) and `v0`, one
`atomicAdd` per element. Add each version as a new kernel and a line in `versions[]`:

| version | change | what it should teach |
|---|---|---|
| v0 | one `atomicAdd` per element (given) | atomics to one address serialise |
| v1 | shared-memory tree, interleaved: `if (tid % (2*s) == 0)` | divergence: half the warps idle, `%` is slow |
| v2 | strided index `index = 2*s*tid` | removes divergence, adds bank conflicts (question 3) |
| v3 | sequential addressing: `for (s = blockDim.x/2; s > 0; s >>= 1) if (tid < s)` | conflict-free |
| v4 | each thread first adds two elements while loading | half the blocks; fewer idle threads in the first step |
| v5 | last 32 elements with `__shfl_down_sync` instead of shared memory | no `__syncthreads` inside one warp |
| v6 | grid-stride loop: a fixed number of blocks (a few per SM), each thread sums many elements, one `atomicAdd` per block | enough work per thread to hide latency |

For each version, record ms, GB/s and % of peak, and one sentence on why it moved. If a
version gets *slower*, that is a finding: explain it.

Profile v1, v3 and v6 locally:

```bash
nvprof --metrics gld_efficiency,shared_load_transactions_per_request,achieved_occupancy ./reduce
```

## Part B: transpose

Copy the harness into `transpose.cu` for a 4096 x 4096 float matrix. The ceiling is a
plain copy kernel (read and write, both coalesced), so measure that first.

| version | change |
|---|---|
| copy | `out[i] = in[i]`: the ceiling |
| naive | `out[x * n + y] = in[y * n + x]`: reads coalesced, writes strided |
| tiled | 32 x 32 tile through shared memory, so reads and writes are both coalesced |
| padded | `__shared__ float tile[32][33]`: removes the bank conflicts |

Bandwidth here counts both read and write: `2 * n * n * 4 bytes / time`.

## Deliverable: `RESULTS.md`

- Your answers to the three questions, then the measured numbers next to them.
- Two tables (reduction, transpose): version, ms, GB/s, % of peak, why it changed.
- One chart: % of peak per version.
