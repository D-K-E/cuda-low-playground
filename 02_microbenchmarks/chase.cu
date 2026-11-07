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

template <typename T> cuptr<T> make_cuptr(size_t n) {
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
  auto d_sink = make_cuptr<unsigned>(1);
  auto d_cycles = make_cuptr<long long>(1);

  std::mt19937 rng(42);
  const int steps = 1 << 20;
  std::printf("%10s %12s %10s\n", "size KB", "cycles/load", "ns/load");

  //
  auto max_as_byte = max_mb * 1024 * 1024;
  for (size_t bytes = 4096; bytes <= max_as_byte; bytes *= 2) {
    size_t nb_els = bytes / sizeof(unsigned);

    // Sattolo's algorthim, visit every element before repeating
    // and don't let hardware guess the next one

    std::vector<unsigned> next(n);
    for (size_t i = 0; i < n; i++) {
      next[i] = static_cast<unsigned>(i);
    }
    for (size_t i = n - 1; i > 0; --i) {
      std::uniform_int_distribution<size_t> dist(0, i - 1);
      size_t j = dist(rng);
      std::swap(next[i], next[j]);
    }
    auto d_next = make_cuptr<unsigned>(nb_els);
    cudaMemcpy(d_next.get(), next.data(), bytes, cudaMemcpyHostToDevice);
    //
    chase<<<1, 1>>>(d_next.get(), steps, d_sink.get()); // warm up
    //
    chase<<<1, 1>>>(d_next.get(), steps, d_sink.get(), d_cycles.get());

    long long cycles = NULL;
    cudaMemcpy(&cycles, d_cycles.get(), sizeof(cycles), cudaMemcpyDeviceToHost);

    double per = static_cast<double>(cycles) / steps;
    std::printf("%10zu %12.1f %10.1f\n", bytes / 1024, per,
                per / (clock_khz / 1e6));
  }
}
