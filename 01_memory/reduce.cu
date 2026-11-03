// reduce: sum n floats; times each version and prints effective bandwidth.
// Usage: ./reduce [n]   (default 2^24)
#include "../common/check.cuh"
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

  // warp level sum, so that we have 1 atomic add for block
  // idea is if we can sum all lanes inside a warp, we reduce the number of
  // ops to do sum
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
