// devinfo: the numbers later modules compare against (peak bandwidth, SM count, limits).
#include "../common/check.cuh"

int main() {
  int dev = 0;
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, dev));
  int sm_clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev));

  std::printf("name                 %s\n", p.name);
  std::printf("compute capability   %d.%d\n", p.major, p.minor);
  std::printf("SMs                  %d\n", p.multiProcessorCount);
  std::printf("SM clock             %.0f MHz\n", sm_clock_khz / 1e3);
  std::printf("global memory        %.1f GB\n", p.totalGlobalMem / 1e9);
  std::printf("peak DRAM bandwidth  %.1f GB/s\n", peak_dram_gbs(dev));
  std::printf("L2 cache             %d KB\n", p.l2CacheSize / 1024);
  std::printf("shared mem per block %zu KB\n", p.sharedMemPerBlock / 1024);
  std::printf("shared mem per SM    %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
  std::printf("registers per SM     %d\n", p.regsPerMultiprocessor);
  std::printf("max threads per SM   %d\n", p.maxThreadsPerMultiProcessor);
  std::printf("warp size            %d\n", p.warpSize);
  std::printf("cooperative launch   %s\n", p.cooperativeLaunch ? "yes" : "no");
}
