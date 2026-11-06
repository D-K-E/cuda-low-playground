// cost of launching tiny kernels vs one cuda graph

#include <chrono>
#include <cstdlib>

__global__ void empty_kernel() {}

namespace ch = std::chrono;

static double now_micro() {
  double m = ch::duration<double, std::micro>(
                 ch::steady_clock::now().time_since_epoch())
                 .count();
  return m;
}

class StreamProxy {
public:
  StreamProxy() { cudaStreamCreate(&s); }
  ~StreamProxy() { cudaStreamDestroy(s); }
  cudaStream_t operator*() { return s; }
  cudaError_t begin_capture(cudaStreamCaptureMode mode) {
    cudaStreamBeginCapture(s, mode);
  }
  cudaError_t end_capture(cudaGraph_t g) { return cudaStreamEndCapture(s, &g); }
  cudaError_t sync() {
    auto err = cudaStreamSynchronize(s);
    return err;
  }

private:
  cudaStream_t s;
};

class GraphProxy {
public:
  GraphProxy(cudaGraphExec_t &exec, unsigned flags) {
    cudaGraphInstantiateWithFlags(&exec, g, flags);
  }
  GraphProxy(cudaGraph_t &g_, cudaGraphExec_t &exec, unsigned flags) : g(g_) {
    cudaGraphInstantiateWithFlags(&exec, g, flags);
  }
  ~GraphProxy() { cudaGraphDestroy(g); }
  cudaGraph_t operator*() { return g; }

private:
  cudaGraph_t g;
};

class GraphExecProxy {
public:
  GraphExecProxy() = default;
  ~GraphExecProxy() { cudaGraphExecDestroy(exec); }
  cudaGraphExec_t operator*() { return exec; }

private:
  cudaGraphExec_t exec;
};

class EventProxy {
public:
  EventProxy() { cudaEventCreate(&e); }
  ~EventProxy() { cudaEventDestroy(e); }

  cudaError_t record(cudaStream_t s) { cudaEventRecord(e, s); }
  cudaError_t record(const StreamProxy &s) { cudaEventRecord(e, *s); }

  cudaError_t record(cudaStream_t s, unsigned int flags) {
    cudaEventRecordWithFlags(e, s, flags);
  }
  cudaError_t record(const StreamProxy &s, unsigned int flags) {
    cudaEventRecordWithFlags(e, *s, flags);
  }
  float elapsed_ms(cudaEvent_t end) {
    float time;
    cudaEventElapsedTime(&time, e, end);
    return time;
  }

  float elapsed_ms(const EventProxy &end) {
    float time;
    cudaEventElapsedTime(&time, e, *end);
    return time;
  }

private:
  cudaEvent_t e;
};

int main(int argc, char **argv) {
  int k = argc > 1 ? std::atoi(argv[1]) : 10000;

  StreamProxy stream_p{};

  empty_kernel<<<1, 1, 0, *stream_p>>>();
  stream_p.sync();

  //
  EventProxy kernel_start{};
  EventProxy kernel_end{};
  kernel_start.record(stream_p);
  for (size_t i = 0; i < k; i++) {
    empty_kernel<<<1, 1, 0, *stream_p>>>();
  }
  kernel_end.record(stream_p);
  float t1 = kernel_start.elapsed_ms(kernel_end);

  EventProxy sync_start{};
  EventProxy sync_end{};
  sync_start.record(stream_p);
  cudaStreamSynchronize(s);
  sync_end.record(stream_p);
  float t2 = sync_end.elapsed_ms(kernel_start);
  std::printf(
      "stream, %d kernels: submit %.2f us/kernel, end to end %.2f us/kernel\n",
      k, t1 / k, t2 / k);

  // 2. launch and wait for each one: the round trip
  const int rt = 1000;
  kernel_start.record(stream_p);
  for (int i = 0; i < rt; i++) {
    empty_kernel<<<1, 1, 0, s>>>();
    cudaStreamSynchronize(*stream_p);
  }
  kernel_end.record(stream_p);
  float t1 = kernel_start.elapsed_ms(kernel_end);
  std::printf("launch + sync each:   %.2f us/kernel\n", t1 / rt);

  // 3. same kernels captured once into graph launched as one unit
  stream_p.begin_capture(cudaStreamCaptureMode);
  for (size_t i = 0; i < k; i++) {
    empty_kernel<<<1, 1, 0, *stream_p>>>();
  }
  cudaGraph_t graph;
  stream_p.end_capture(graph);
  //

  //
  GraphExecProxy exec{};
  kernel_start.record(stream_p);
  GraphProxy proxy{graph, exec, 0};
  kernel_end.record(stream_p);
  t1 = kernel_start.elapsed_ms(kernel_end);
  std::printf("graph instantiate:    %.0f us total\n", t1);

  //
  cudaGraphLaunch(*exec, *stream_p); // warm up
  cudaStreamSynchronize(*stream_p);
  sync_start.record(stream_p);
  cudaGraphLaunch(*exec, *stream_p);
  cudaStreamSynchronize(*stream_p);
  sync_end.record(stream_p);
  t1 = sync_start.elapsed_ms(sync_end);

  std::printf("graph, %d kernels:    end to end %.2f us/kernel\n", k, t1 / k);

  // TODO: sweep kernels per graph (1, 10, 100, 1000, 10000) and plot us/kernel
  // for the stream and the graph. Then give the kernel real work (a loop of N
  // FMAs) and find the kernel duration at which launch overhead stops
  // mattering.
}
