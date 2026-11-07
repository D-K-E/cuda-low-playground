# 00 Setup: what each machine can actually do

**Goal:** a table of which tools and features work on the local Maxwell card and on the
cloud GPU, verified rather than assumed. Every later module refers to it.

**CV line:** none; this is groundwork.

## Steps

1. On the local machine:
   ```bash
   nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
   nvcc --version
   make && ./devinfo
   ```
   Note the exact model, the driver version and the CUDA version. Questions:
   - Which compute capability does `devinfo` report: 5.0 (M2000M) or 5.2 (M4000/M5000)?
     Update the hardware table in `../README.md` to match.
   - Is the toolkit 12.x? If it's 13.x, `-arch=sm_50` fails: install 12.x next to it.
   - Does `nvprof --version` work? If not, install CUDA 11.8 (supports Maxwell and ships
     `nvprof`) for the profiling steps.
   - The peak bandwidth `devinfo` prints: does it match the card's datasheet?
2. Try each tool on a trivial kernel (`devinfo` will do) and fill in the table below:
   ```bash
   compute-sanitizer ./devinfo
   cuda-gdb --batch -ex run ./devinfo
   nvprof ./devinfo
   nsys profile -o devinfo ./devinfo
   ncu ./devinfo          # expected to refuse on Maxwell
   ```
3. Repeat on the cloud GPU at the start of the first cloud session (`make ARCH=sm_80`).

## Deliverable: `RESULTS.md`

| tool / feature | local `sm_50` | cloud `sm_8x` |
|---|---|---|
| GPU model, CUDA version, driver | tesla T4 | |
| peak DRAM GB/s (devinfo) | 320.1 Gb/s| |
| compute-sanitizer | | |
| cuda-gdb | | |
| nvprof | | |
| Nsight Systems (CUDA trace) | | |
| Nsight Compute | | |
| CUB / Thrust / cuBLAS | | |
| cuDNN (which version) | | |
| Triton | | |
| CUDA graphs | | |
