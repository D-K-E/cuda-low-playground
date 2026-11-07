# 08 Cluster networking: linking GPUs together

**Goal:** be able to draw a GPU training cluster and explain where each byte of an
all-reduce travels: NVLink inside a node, InfiniBand or RoCE between nodes, GPUDirect RDMA
from GPU memory to the NIC, rail-optimised switch layouts, and PXN. Back it with
`nccl-tests` measurements from a rented node.

**CV line:** "Wrote a design study of GPU cluster interconnects (NVLink/NVSwitch,
InfiniBand vs RoCE, GPUDirect RDMA, rail-optimised fabrics, PXN) and measured intra-node
NCCL bandwidth over NVLink vs PCIe." Claim only the measurements you actually ran; the rest
is "design study".

**Read:** NVIDIA DGX SuperPOD reference architecture (the network section: rails, leaf
and spine). The NCCL docs on `NCCL_PXN_DISABLE`, `NCCL_P2P_LEVEL`, `NCCL_NET_GDR_LEVEL`. The
NVIDIA GPUDirect RDMA docs (overview). One RoCE overview covering PFC and ECN (why RoCE
needs a lossless or congestion-controlled network).

## Concepts, each as a question to answer in your notes

1. **NVLink / NVSwitch:** what bandwidth per GPU, and why does an all-reduce inside an
   NVSwitch node not care about ring order?
2. **PCIe topology:** GPU, PCIe switch, CPU socket, NIC. What does `nvidia-smi topo -m` mean
   by `NV#`, `PIX`, `PXB`, `PHB`, `SYS`?
3. **GPUDirect P2P and RDMA:** what copy does each one remove? Draw the data path with and
   without it.
4. **InfiniBand vs RoCE:** same RDMA verbs, different link layer. What does RoCE need from
   the Ethernet network to perform (PFC, ECN, DCQCN), and what goes wrong without it?
5. **Rail-optimised fabric:** in an 8-GPU node with 8 NICs, GPU *k* of every node connects
   to leaf switch *k*. Why does that keep most all-reduce traffic to one switch hop?
6. **PXN (PCIe x NVLink):** a GPU sends to a remote node through a *different* GPU's NIC,
   moving the data over NVLink first. When does that help (rail alignment, fewer flows)?

## Steps

1. Draw (by hand or as an SVG) one 8-GPU node and a 4-node rail-optimised cluster. Mark
   bandwidths on every link and the path of one ring all-reduce.
2. On the multi-GPU cloud node from module 07: save `nvidia-smi topo -m`, then run
   `all_reduce_perf` with defaults, with `NCCL_P2P_DISABLE=1` (forces traffic through host
   memory) and, if the node has NVLink, compare. Explain the gap from your link table.
3. If the provider offers 2 nodes with InfiniBand for a short time, run `nccl-tests` across
   them with and without `NCCL_NET_GDR_LEVEL` and `NCCL_PXN_DISABLE=1`. Otherwise skip: the
   design notes stand on their own.

## Deliverable: `RESULTS.md`

The diagrams, answers to the six questions, the `topo -m` output with an explanation, and
the NVLink vs no-P2P bandwidth table.
