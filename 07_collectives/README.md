# 07 Collectives: the algorithms behind distributed training

**Goal:** implement ring and tree all-reduce yourself (MPI, runs on any machine), predict
their time with a cost model, then compare the model with NCCL on a multi-GPU node.

**CV line:** "Implemented ring and tree all-reduce with MPI, built an alpha-beta cost model
for them, and validated it against NCCL measurements on an 8-GPU NVLink node."

**Read:** Thakur, Rabenseifner and Gropp, *Optimization of Collective Communication
Operations in MPICH* (the cost models). The NCCL docs, *Collective Operations* and the
environment variables section (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_DEBUG`). Optional:
Patarasuk and Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of
Workstations* (why ring is bandwidth-optimal).

## The model before the code

Sending `n` bytes costs `alpha + n * beta` (alpha = latency per message, beta = time per byte).
With `p` ranks and a message of `n` bytes:

| algorithm | steps | time | good for |
|---|---|---|---|
| ring all-reduce (reduce-scatter + all-gather) | `2(p-1)` | `2(p-1) alpha + 2 (p-1)/p * n * beta` | large `n`: bandwidth term is near-optimal |
| binary tree (reduce + broadcast) | `2 log2 p` | `2 log2(p) (alpha + n * beta)` | small `n`: fewer steps |
| recursive doubling | `log2 p` | `log2(p) (alpha + n * beta)` | small `n`, `p` a power of 2 |

Questions: at what `n` does ring overtake tree for `p = 8`? Why does NCCL use a *double
binary tree* across nodes? What does the `(p-1)/p` factor mean in practice ("bus bandwidth"
in `nccl-tests`)?

## Steps

1. **CPU, MPI** (this machine is fine: `mpirun -np 4` on one host). `allreduce.cpp` has the
   harness (size sweep, exact check, timing) and `MPI_Allreduce`; fill in `ring` and
   `recursive_doubling` with `MPI_Sendrecv`. `make run` builds and runs on 4 ranks.
   Sweep `n` from 8 bytes to 64 MB and plot time vs `n` for yours and the MPI library's.
2. Fit alpha and beta from a ping-pong test, then overlay the model on your measurements.
3. **Multi-GPU cloud node:** run `nccl-tests` (`all_reduce_perf -b 8 -e 1G -f 2 -g 8`) with
   `NCCL_ALGO=Ring` and `NCCL_ALGO=Tree`. Read `NCCL_DEBUG=INFO` to see the topology NCCL
   detected (rings, trees, NVLink vs PCIe). Compare algorithm bandwidth and bus bandwidth with
   your model. Module 08 continues from this run.
4. Optional: the ring all-reduce with NCCL point-to-point (`ncclSend`/`ncclRecv`) on the GPUs.

## Deliverable: `RESULTS.md`

The cost-model table with your fitted alpha and beta, the time-vs-size chart (your MPI
versions, MPI library, model), the NCCL Ring vs Tree chart, and where the crossover is.
