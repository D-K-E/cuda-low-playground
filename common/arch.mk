# Picks ARCH (e.g. sm_75) from the first visible GPU; override with `make ARCH=sm_80`.
# Falls back to sm_50 (the local Maxwell card) when nvidia-smi is missing or too old.
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r' | sed 's/^\(..*\)$$/sm_\1/')
ARCH ?= $(if $(DETECTED_ARCH),$(DETECTED_ARCH),sm_50)
