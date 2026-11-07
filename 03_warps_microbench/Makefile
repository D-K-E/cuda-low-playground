include ../common/arch.mk
NVCCFLAGS := -O3 -arch=$(ARCH) -lineinfo -std=c++17
BINS := launch chase

all: $(BINS)

%: %.cu ../common/check.cuh
	nvcc $(NVCCFLAGS) $< -o $@

# Maxwell caches global loads in L2 only by default; this build also caches them in L1.
chase-l1: chase.cu ../common/check.cuh
	nvcc $(NVCCFLAGS) -Xptxas -dlcm=ca $< -o $@

clean:
	rm -f $(BINS) chase-l1

.PHONY: all clean
