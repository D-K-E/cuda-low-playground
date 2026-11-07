// allreduce: ring and recursive-doubling all-reduce over MPI, checked against and timed
// next to MPI_Allreduce, for message sizes from 8 bytes to 64 MB.
// make && mpirun -np 4 ./allreduce
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <vector>

// Returns false while not implemented; the harness then skips it.
using Impl = bool (*)(std::vector<float>& buf, MPI_Comm comm);

static bool library(std::vector<float>& buf, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, buf.data(), int(buf.size()), MPI_FLOAT, MPI_SUM, comm);
  return true;
}

// TODO ring all-reduce. Split buf into p chunks (chunk c is [c*n/p, (c+1)*n/p)).
// Reduce-scatter, p-1 steps: at step s, send chunk (rank - s) mod p to rank + 1, receive
// chunk (rank - s - 1) mod p from rank - 1 and add it in (one MPI_Sendrecv per step).
// Afterwards this rank holds the fully reduced chunk (rank + 1) mod p.
// All-gather, p-1 more steps: pass the reduced chunks around the ring the same way,
// replacing instead of adding.
static bool ring(std::vector<float>& buf, MPI_Comm comm) {
  (void)buf;
  (void)comm;
  return false;
}

// TODO recursive doubling (p a power of 2): log2(p) steps; at step k, exchange the whole
// buffer with rank ^ (1 << k) and add what you receive.
static bool recursive_doubling(std::vector<float>& buf, MPI_Comm comm) {
  (void)buf;
  (void)comm;
  return false;
}

struct Version {
  const char* name;
  Impl run;
};

static const Version versions[] = {
    {"MPI_Allreduce", library},
    {"ring", ring},
    {"recursive doubling", recursive_doubling},
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, p = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  if (rank == 0) std::printf("%d ranks\n%-20s %12s %12s\n", p, "version", "bytes", "us");

  for (const Version& v : versions) {
    for (size_t n = 2; n <= (size_t(64) << 20) / sizeof(float); n *= 4) {
      // Small integers, so every sum is exact and the check can use ==.
      std::vector<float> buf(n);
      auto fill = [&] {
        for (size_t i = 0; i < n; ++i) buf[i] = float(rank + 1 + int(i % 3));
      };

      fill();
      if (!v.run(buf, MPI_COMM_WORLD)) {
        if (rank == 0) std::printf("%-20s not implemented\n", v.name);
        break;
      }
      int ok = 1;
      for (size_t i = 0; i < n; ++i)
        if (buf[i] != float(p * (p + 1) / 2 + p * int(i % 3))) ok = 0;
      MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

      // More repetitions for small messages, where one run is a few microseconds.
      int reps = int(std::clamp<size_t>((size_t(1) << 24) / n, 3, 1000));
      double total = 0;
      for (int r = 0; r < reps; ++r) {
        fill();
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        v.run(buf, MPI_COMM_WORLD);
        total += MPI_Wtime() - t0;
      }
      // The collective is as slow as its slowest rank.
      MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (rank == 0)
        std::printf("%-20s %12zu %12.2f %s\n", v.name, n * sizeof(float), 1e6 * total / reps,
                    ok ? "" : "WRONG");
    }
  }
  MPI_Finalize();
}
