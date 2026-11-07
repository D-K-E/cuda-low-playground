include ../common/arch.mk
NVCCFLAGS := -O3 -arch=$(ARCH) -lineinfo -std=c++17

devinfo: devinfo.cu ../common/check.cuh
	nvcc $(NVCCFLAGS) $< -o $@

clean:
	rm -f devinfo

.PHONY: clean
