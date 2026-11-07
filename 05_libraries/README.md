# 05 Libraries: when to use which

**Goal:** solve the same few problems with your own kernels and with the libraries, so you
can say from measurements when a library wins, when it doesn't, and what it does inside.

**CV line:** "Benchmarked hand-written CUDA kernels against CUB, Thrust, cuBLAS, cuDNN,
CUTLASS and Triton for scan, sort, reduction, GEMM, convolution and fused softmax."

**Read:** the CUB docs (block-, warp- and device-wide primitives), the Thrust quick start,
the Triton tutorials 01-03 (vector add, fused softmax, matmul).

## Local (`sm_50`)

| problem | yours | library | question |
|---|---|---|---|
| reduction | module 01 v6 | `cub::DeviceReduce::Sum`, `thrust::reduce` | how close is your best version? what does CUB's temp-storage API cost you? |
| prefix sum | write a block scan, then a multi-block scan | `cub::DeviceScan::ExclusiveSum` | why is a single-pass scan hard (look up "decoupled look-back")? |
| sort 2^24 keys | none | `thrust::sort`, `cub::DeviceRadixSort` | keys/s for 32-bit vs 64-bit keys |
| inside a kernel | your module 01 shuffle reduce | `cub::WarpReduce`, `cub::BlockReduce` | same SASS? (module 02) |
| GEMM | module 04 | cuBLAS | from module 04 |
| 3x3 convolution | naive kernel | cuDNN (if module 00 says it runs) | which algorithm does `cudnnFind...` pick and why? |

## Cloud

| problem | yours | library |
|---|---|---|
| fused softmax over rows | CUDA kernel, one warp per row | Triton tutorial 02, `torch.softmax` |
| FP16 GEMM | module 04 Tensor Core version | Triton tutorial 03, CUTLASS, cuBLAS |

For Triton, also read the generated PTX (`kernel.asm["ptx"]`) and compare it with yours.

## Deliverable: `RESULTS.md`

A table per problem with throughput for each implementation, and a closing section:
"which library when" in five or six bullet points, each backed by a number above.
