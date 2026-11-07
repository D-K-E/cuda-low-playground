#pragma once
#include <cstdio>
#include <cstdlib>

// Wrap every CUDA runtime call; after a kernel launch use CHECK(cudaGetLastError()).
#define CHECK(call)                                                        \
  do {                                                                     \
    cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(e_));                                \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// Peak DRAM bandwidth in GB/s from the device attributes: clock (kHz) x 2 (double
// data rate) x bus width (bits) / 8.
inline double peak_dram_gbs(int dev = 0) {
  int clock_khz = 0, bus_bits = 0;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
  CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  return 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}
