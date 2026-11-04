// reduce: sum n floats; times each version and prints effective bandwidth.
// Usage: ./reduce [n]   (default 2^24)
#include "../common/check.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
#include <vector>

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
__global__ void reduce_v6(const float *in, float *out, int n) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x; // cover whole grid
  float s = 0.f;
  for (size_t i = gtid; i < n; i += stride) {
    s += in[i];
  }
  const int warp_size = 32;
  //
  unsigned full_mask = __ballot(1);
  for (size_t warp_offset = warp_size / 2; warp_offset > 0; warp_offset /= 2) {
    s += __shfl_down_sync(full_mask, sum, warp_offset);
  }

  // warp level sum, so that we have 1 atomic add for block
  int lane = threadIdx.x;
  while (lane > warp_size) {
    lane -= warp_size;
  }
  const int warp_id = threadIdx.x / warp_size;
  __shared__ float warp_sums[warp_size];
  if (lane == 0) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int num_warps = blockDim.x / warp_size;
    float block_sum = lane < num_warps ? warp_sums[lane] : 0.f;
    unsigned full_mask = __ballot(1);
    for (size_t warp_offset = warp_size / 2; warp_offset > 0;
         warp_offset /= 2) {
      block_sum += __shfl_down_sync(full_mask, block_sum, warp_offset);
    }
    if (lane == 0) {
      atomicAdd(out, block_sum);
    }
  }
}

namespace cg = cooperative_groups;

template <size_t NbWarpPerBlock>
__global__ void reduce_v7(const float *in, float *out, int n) {
    // shape of this algorithm is the following

    // grid stride sum accumulates elements across stride to a thread **register**

    // then we accumulate each thread register to warp leader (first
    // lane/thread)

    // then we store each warp leader value to shared memory
    // shared memory size is equal to the number of warps per block
    // so each warp leader has a slot there

    // then reduce values accumulated in the shared memory into a single
    // value, at this point the value represents the whole block

    // then we add that value representing the block to out via atomicAdd 
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
static void run_v6(const float *in, float *out, int n) {
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
