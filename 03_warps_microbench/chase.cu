// chase: latency of one dependent global load, for array sizes from 4 KB upwards.
// Each load's address is the previous load's result, so loads can't overlap: the time
// per step is the latency of whichever level (L1, L2, DRAM) holds the array.
// Usage: ./chase [max_mb]   (default 256)
#include "../common/check.cuh"
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

__global__ void chase(const unsigned* next, int steps, unsigned* sink, long long* cycles) {
  unsigned j = 0;
  long long t0 = clock64();
  for (int i = 0; i < steps; ++i) j = next[j];
  long long t1 = clock64();
  *sink = j;  // without this the compiler deletes the loop
  *cycles = t1 - t0;
}

int main(int argc, char** argv) {
  size_t max_mb = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 256;
  int clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));

  unsigned* d_sink;
  long long* d_cycles;
  CHECK(cudaMalloc(&d_sink, sizeof(unsigned)));
  CHECK(cudaMalloc(&d_cycles, sizeof(long long)));
  std::mt19937 rng(42);
  const int steps = 1 << 20;

  std::printf("%10s %12s %10s\n", "size KB", "cycles/load", "ns/load");
  for (size_t bytes = 4096; bytes <= max_mb << 20; bytes *= 2) {
    size_t n = bytes / sizeof(unsigned);
    // Sattolo's algorithm: a random permutation that is one single cycle, so the chase
    // visits every element before repeating and the hardware can't predict the next one.
    std::vector<unsigned> next(n);
    for (size_t i = 0; i < n; ++i) next[i] = unsigned(i);
    for (size_t i = n - 1; i > 0; --i) {
      std::uniform_int_distribution<size_t> pick(0, i - 1);
      std::swap(next[i], next[pick(rng)]);
    }

    unsigned* d_next;
    CHECK(cudaMalloc(&d_next, bytes));
    CHECK(cudaMemcpy(d_next, next.data(), bytes, cudaMemcpyHostToDevice));
    chase<<<1, 1>>>(d_next, steps, d_sink, d_cycles);  // warm-up: fills the caches
    CHECK(cudaGetLastError());
    chase<<<1, 1>>>(d_next, steps, d_sink, d_cycles);
    CHECK(cudaGetLastError());
    long long cycles = 0;
    CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_next));

    double per = double(cycles) / steps;
    std::printf("%10zu %12.1f %10.1f\n", bytes / 1024, per, per / (clock_khz / 1e6));
  }
  CHECK(cudaFree(d_sink));
  CHECK(cudaFree(d_cycles));
}
