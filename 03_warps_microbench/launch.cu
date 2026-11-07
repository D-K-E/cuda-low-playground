// launch: cost of launching tiny kernels one at a time vs as one CUDA graph.
// Usage: ./launch [kernels]   (default 10000)
#include "../common/check.cuh"
#include <chrono>
#include <cstdlib>

__global__ void empty_kernel() {}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  int k = argc > 1 ? std::atoi(argv[1]) : 10000;
  cudaStream_t s;
  CHECK(cudaStreamCreate(&s));

  // The first launch loads the module and is much slower: keep it out of the numbers.
  empty_kernel<<<1, 1, 0, s>>>();
  CHECK(cudaGetLastError());
  CHECK(cudaStreamSynchronize(s));

  // 1. k launches back to back, one wait at the end.
  double t0 = now_us();
  for (int i = 0; i < k; ++i) empty_kernel<<<1, 1, 0, s>>>();
  double t1 = now_us();  // the host has queued them; the GPU may still be running
  CHECK(cudaStreamSynchronize(s));
  double t2 = now_us();
  std::printf("stream, %d kernels: submit %.2f us/kernel, end to end %.2f us/kernel\n", k,
              (t1 - t0) / k, (t2 - t0) / k);

  // 2. launch and wait for each one: the round trip.
  const int rt = 1000;
  t0 = now_us();
  for (int i = 0; i < rt; ++i) {
    empty_kernel<<<1, 1, 0, s>>>();
    CHECK(cudaStreamSynchronize(s));
  }
  t1 = now_us();
  std::printf("launch + sync each:   %.2f us/kernel\n", (t1 - t0) / rt);

  // 3. The same k kernels captured once into a graph, then launched as one unit.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < k; ++i) empty_kernel<<<1, 1, 0, s>>>();
  CHECK(cudaStreamEndCapture(s, &graph));
  t0 = now_us();
  CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));  // works on CUDA 11.4+ and 12
  t1 = now_us();
  std::printf("graph instantiate:    %.0f us total\n", t1 - t0);

  CHECK(cudaGraphLaunch(exec, s));  // warm-up
  CHECK(cudaStreamSynchronize(s));
  t0 = now_us();
  CHECK(cudaGraphLaunch(exec, s));
  CHECK(cudaStreamSynchronize(s));
  t1 = now_us();
  std::printf("graph, %d kernels:    end to end %.2f us/kernel\n", k, (t1 - t0) / k);

  // TODO: sweep kernels per graph (1, 10, 100, 1000, 10000) and plot us/kernel for the
  // stream and the graph. Then give the kernel real work (a loop of N FMAs) and find the
  // kernel duration at which launch overhead stops mattering.

  CHECK(cudaGraphExecDestroy(exec));
  CHECK(cudaGraphDestroy(graph));
  CHECK(cudaStreamDestroy(s));
}
