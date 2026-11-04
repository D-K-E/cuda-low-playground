# Miscellaneous work on CUDA Programming

Various programming experiments with CUDA for deepening various concepts.

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

- Streaming Multiprocessor (SM): hardware unit that houses blocks and warps
- SM Register: total number registers assigned to an SM. This is GPU
  architecture dependent. 64k registers are typical, meaning 64000*32 bit
  registers. 


## Tips and Tricks

1. Warp reduction, see `01_memory/reduce.cu`. The shape is similar for most
problems so it pays to know it:
  1. Move relevant values to registers.
  2. Do something with those values in registers using warp level primitives,
  save your result to warp leader (first lane)
  3. Store each warp leader to shared memory, so that they can be shared at
  block level. The size of the shared memory should be equal to the number of
  warps in a block, so that we can house the result of each warp.
  4. Synchronize at this point, so that each thread in the block has a value
  to work with in the shared memory
  5. Compute what you need with shared memory
  6. Store result.

