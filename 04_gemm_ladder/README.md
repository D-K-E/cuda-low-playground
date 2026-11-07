# 04 GEMM ladder: from naive to Tensor Cores

**Goal:** write single-precision GEMM from naive to register-blocked on Maxwell, then FP16
GEMM with Tensor Cores and asynchronous loads on the cloud GPU, measuring each step
against cuBLAS (and CUTLASS on the cloud GPU).

**CV line:** "Implemented a CUDA GEMM reaching X% of cuBLAS SGEMM on Maxwell and a Tensor
Core (`mma.sync` + `cp.async` pipeline) GEMM reaching Y% of cuBLAS on Ampere."

**Read:** Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
(blog post, the main guide here). CUTLASS docs, *Efficient GEMM in CUDA*. *PTX ISA*,
the `mma` section, for the Tensor Core part.

## Before any code

1. GEMM of two `N x N` matrices does `2 N^3` flops and must move at least `3 N^2` floats.
   For `N = 4096`: arithmetic intensity (flops per byte)? Compare with your GPU's peak
   FLOP/s divided by peak bandwidth (Maxwell has no FMA throughput listed in `devinfo`:
   SMs x 128 FP32 lanes x 2 x clock). Is GEMM compute-bound or memory-bound?
2. Naive GEMM: each thread computes one output and reads a row and a column. How many
   bytes does each thread load per flop? How much of that does a 32 x 32 shared tile save?

## Local: SIMT ladder (`sm_50`, FP32)

| step | kernel | expected lesson |
|---|---|---|
| 0 | cuBLAS `cublasSgemm` | the target |
| 1 | naive, one output per thread | global memory bound |
| 2 | coalesced (swap thread index order) | the same as module 01, on a new kernel |
| 3 | shared-memory tiles | data reuse |
| 4 | 1D register blocking (each thread computes a column of 8 outputs) | more FMAs per shared load |
| 5 | 2D register blocking (8 x 8 outputs per thread) | most of the gap to cuBLAS |
| 6 | vectorised `float4` loads, transposed A tile | fewer load instructions (module 02) |

Check every version against cuBLAS for correctness (max relative error), not only speed.

## Cloud: Tensor Core ladder (`sm_80`+, FP16 in, FP32 accumulate)

| step | kernel |
|---|---|
| 0 | cuBLAS `cublasGemmEx` with FP16 inputs, and a CUTLASS device GEMM |
| 1 | WMMA API (`nvcuda::wmma`), one warp per 16 x 16 tile |
| 2 | shared-memory tiles feeding WMMA |
| 3 | `cp.async` double buffering: load the next tile while computing this one |
| 4 | optional: inline-PTX `mma.sync.m16n8k16` and `ldmatrix` |

Profile steps 2 and 3 in Nsight Compute: tensor pipe utilisation, and memory stalls before
and after `cp.async`.

## Deliverable: `RESULTS.md`

- GFLOP/s and % of cuBLAS for every step, on both GPUs, as a chart.
- One paragraph per step on why it moved.
- The roofline point of your best kernel on each GPU.
