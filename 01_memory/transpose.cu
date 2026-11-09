// reduce: sum n floats; times each version and prints effective bandwidth.
// Usage: ./reduce [n]   (default 2^24)
#include "../common/check.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
#include <vector>

// v0: one atomicAdd per element. Correct and slow: the baseline.
namespace cg = cooperative_groups;

__global__ void transpose_naive(const float *in, float *out, int width,
                                int height) {
  // assuming grid size matches the matrix size
  // in: n x m
  // out: m x n

  // x coordinate accessible by current thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // y coordinate accessible by current thread
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //
  // in idx
  if (x < width && y < height) {
    int in_idx = y * width + x;
    int out_idx = x * height + y;
    out[out_idx] = in[in_idx];
  }
}

template <size_t Tile>
__global__ void transpose_shared(const float *in, float *out, int width,
                                 int height);

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
