// latency of one dependent global load for array sizes from 4kb
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

__global__ void chase(const unsigned *next, int steps, unsigned *sink,
                      long long *cycles) {
  unsigned j = 0;
  long long t0 = clock64();
  for (int i = 0; i < steps; i++) {
    j = next[j];
  }
  long long t1 = clock64();
  *sink = j;
  *cycles = t1 - t0;
}

template <typename T> struct CuPtrDeleter {
  void operator()(T *p) { cudaFree(p); }
};

template <typename T> using cuptr<T> = std::unique_ptr<T, CuPtrDeleter<T>{}>;

template <typename T> cuptr<T> make_ptr(size_t n) {
  size_t s = n * sizeof(T);
  T *tptr;
  std::unique_ptr<T, CuPtrDeleter<T>> ptr{tptr};
  cudaMalloc(ptr.data(), s);
  return ptr;
}

int main(int argc, char **argv) {
  size_t max_mb = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 256;
  int clock_khz = 0;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);

  //
  auto d_sink = make_ptr<unsigned>(1);
  auto d_cycles = make_ptr<long long>(1);

  std::mt19937 rng(42);
  const int steps = 1 << 20;
  std::printf("%10s %12s %10s\n", "size KB", "cycles/load", "ns/load");

  //
  auto max_as_byte = max_mb * 1024 *1024;
  for (size_t bytes = 4096; bytes <= max_as_byte; bytes *= 2) {
      size_t nb_els = bytes / sizeof(unsigned);
  }
}
