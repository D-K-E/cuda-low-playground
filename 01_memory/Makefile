include ../common/arch.mk
NVCCFLAGS := -O3 -arch=$(ARCH) -lineinfo -std=c++17
BINS := reduce $(if $(wildcard transpose.cu),transpose)

all: $(BINS)

%: %.cu ../common/check.cuh
	nvcc $(NVCCFLAGS) $< -o $@

clean:
	rm -f reduce transpose

.PHONY: all clean
