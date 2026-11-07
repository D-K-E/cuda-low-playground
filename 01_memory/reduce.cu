// reduce: sum n floats; times each version and prints effective bandwidth.
// Usage: ./reduce [n]   (default 2^24)
#include "../common/check.cuh"
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>

// v0: one atomicAdd per element. Correct and slow: the baseline.
__global__ void reduce_v0(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(out, in[i]);
}

template <size_t Shared>
__global__ void reduce_v1(const float *in, float *out, int n) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  __shared__ float shared[Shared];
  if (gtid < n) {
    shared[tid] = in[gtid];
  } else {
    shared[tid] = 0.f;
  }
  __syncthreads();

  //
  for (size_t step = 1; step < blockDim.x; step *= 2) {
    if (gtid % (2 * step) == 0) {
      shared[tid] += shared[tid + step];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared[tid];
  }
}

template <size_t Shared>
__global__ void reduce_v3(const float *in, float *out, int n) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  __shared__ float shared[Shared];
  if (gtid < n) {
    shared[tid] = in[gtid];
  } else {
    shared[tid] = 0.f;
  }
  __syncthreads();

  //
  for (size_t step = blockDim.x; step > 0; step /= 2) {
    if (tid < step) {
      shared[tid] += shared[tid + step];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared[tid];
  }
}

template <size_t MaxNbWarpsPerBlock>
__global__ void reduce_v6(const float *in, float *out, int n) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x; // cover whole grid
  float s = 0.f;
  for (size_t i = gtid; i < n; i += stride) {
    s += in[i];
  }

  // warp level sum, so that we have 1 atomic add for block
  // idea is if we can sum all lanes inside a warp, we reduce the number of
  // ops to do sum
}

namespace cg = cooperative_groups;

template <size_t NbWarpPerBlock>
__global__ void reduce_v7(const float *in, float *out, int n) {
  constexpr int warp_size = 32;
  static_assert(NbWarpPerBlock >= 1 && NbWarpPerBlock <= warp_size,
                "block must be 1..32 warps (max 1024 threads)");
  // shape of this algorithm is the following

  // grid stride sum accumulates elements across stride to a thread **register**
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float s = 0.f;
  for (int i = tid; i < n; i += stride) {
    s += in[i];
  }

  // then we accumulate each thread register to warp leader (first
  // lane/thread)
  auto tg = cg::this_thread_block();
  auto tiled = cg::tiled_partition<warp_size>(tg);

  // reduce
  for (int warp_offset = warp_size / 2; warp_offset > 0; warp_offset /= 2) {
    s += tiled.shfl_down(s, warp_offset);
  }

  // then we store each warp leader value to shared memory
  // shared memory size is equal to the number of warps per block
  // so each warp leader has a slot there
  __shared__ float warp_sums[NbWarpPerBlock];
  int lane = tiled.thread_rank();
  const int warp_id = threadIdx.x / warp_size;
  if (lane == 0) {
    warp_sums[warp_id] = s;
  }
  cg::sync(tg);

  // then reduce values accumulated in the shared memory into a single
  // value, at this point the value represents the whole block
  // that value is the first warp
  if (warp_id == 0) {
    float wsum = 0.f;
    for (size_t k = lane; k < NbWarpPerBlock; k += warp_size) {
      wsum += warp_sums[k];
    }
    for (size_t i = warp_size / 2; i > 0; i /= 2) {
      wsum += tiled.shfl_down(wsum, i);
    }
    // then we add that value representing the block to out via atomicAdd
    if (lane == 0) {
      atomicAdd(out, wsum);
    }
  }
}

template <size_t MaxWarp>
__global__ void reduce_v8(const float *in, float *out, int n) {
  // shape of the algorithm

  // grid stride to store values to registers
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  float s = 0.f;
  for (size_t i = gid; i < n; i += stride) {
    s += in[i];
  }

  // move values from registers to first lane of warp
  auto block = cg::this_thread_block();
  __syncwarp();
  auto active = cg::coalesced_threads();
  const int warp_size = 32;
  for (size_t warp_offset = warp_size / 2; warp_offset > 0; warp_offset /= 2) {
    s += active.shfl_down(s, warp_offset);
  }

  // move from first lane to shared memory. Memory size is number of warps per
  // block
  const int warp_id = threadIdx.x / warp_size;
  int lane = active.thread_rank();
  __shared__ float warp_sums[MaxWarp];
  if (lane == 0) {
    warp_sums[warp_id] = s;
  }
  cg::sync(block);

  // now move sums inside shared memory to the first warp_id
  if (warp_id == 0) {
    const int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    float bsum = 0.f;
    for (size_t i = lane; i < num_warps; i += warp_size) {
      bsum += warp_sums[i];
    }
    auto g = cg::coalesced_threads();
    for (size_t i = warp_size / 2; i > 0; i /= 2) {
      bsum += g.shfl_down(bsum, i);
    }
    if (lane == 0) {
      atomicAdd(out, bsum);
    }
  }
}

template <size_t NbWarpPerBlock>
__global__ void reduce_v8b(const float *in, float *out, int n) {
  // shape of the algorithm
  const int warp_size = 32;
  static_assert(NbWarpPerBlock >= 1 && NbWarpPerBlock <= warp_size,
                "block must be 1..32 warps (max 1024 threads)");

  // load to registers with grid stride
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float s = 0.f;
  for (size_t i = tid; i < n; i += stride) {
    s += in[i];
  }

  // sum register values at the warp level
  auto block = cg::this_thread_block();
  auto g = cg::tiled_partition<warp_size>(block);
  for (size_t warp_offset = warp_size / 2; warp_offset > 0; warp_offset /= 2) {
    s += g.shfl_down(s, warp_offset);
  }

  // assign warp sum to a shared buffer that can be accessed by each block's
  // warps
  __shared__ float warp_sums[NbWarpPerBlock];
  int lane = g.thread_rank();
  int warp_id = threadIdx.x / warp_size;
  if (lane == 0) {
    warp_sums[warp_id] = s;
  }
  cg::sync(block);

  // sum the values in shared buffer
  if (warp_id == 0) {
    float bsum = 0.f;
    int stride = warp_size;
    int warp_nb = (blockDim.x + warp_size -1) / warp_size;

    // each warp lane contains the partial sum
    for (size_t woffset = lane; woffset < warp_nb; woffset += stride) {
      bsum += warp_sums[woffset];
    }

    for (size_t i = warp_size / 2; i > 0; i /= 2) {
      bsum += g.shfl_down(bsum, i);
    }

    // sum the warp lanes again
    if (lane == 0) {
      atomicAdd(out, bsum);
    }
  }

  // write it per block
}

// TODO v1..v6: see README.md. Each is a kernel plus a run_vN launcher below.

using Launch = void (*)(const float *in, float *out, int n);
struct Version {
  const char *name;
  Launch run;
};

static void run_v0(const float *in, float *out, int n) {
  reduce_v0<<<(n + 255) / 256, 256>>>(in, out, n);
}
static void run_v1(const float *in, float *out, int n) {
  constexpr size_t shared = 32;
  reduce_v1<shared><<<(n + 255) / 256, 256>>>(in, out, n);
}

static void run_v3(const float *in, float *out, int n) {
  constexpr size_t shared = 32;
  reduce_v3<shared><<<(n + 255) / 256, 256>>>(in, out, n);
}

static Version versions[] = {
    {"v0 atomic per element", run_v0},
};

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 1 << 24;

  // Values -3..3 repeating: every partial sum is a small integer, so float
  // addition is exact in any order and the result can be compared with ==.
  std::vector<float> h(n);
  double want = 0;
  for (int i = 0; i < n; ++i) {
    h[i] = float(i % 7) - 3.0f;
    want += h[i];
  }

  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CHECK(cudaMalloc(&d_out, sizeof(float)));
  CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  double peak = peak_dram_gbs();
  std::printf("n = %d, peak %.1f GB/s\n", n, peak);

  for (const Version &v : versions) {
    // One untimed run: checks the result and warms up.
    float got = 0;
    CHECK(cudaMemset(d_out, 0, sizeof(float)));
    v.run(d_in, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // Timed runs. The memset is included; it is tiny next to reading n floats.
    const int reps = 20;
    CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
      CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
      v.run(d_in, d_out, n);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= reps;

    double gbs = n * sizeof(float) / (ms * 1e6);
    std::printf("%-28s %9.3f ms %8.1f GB/s %5.1f%% of peak  %s\n", v.name, ms,
                gbs, 100 * gbs / peak, got == want ? "ok" : "WRONG");
  }

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
}
