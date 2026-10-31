NVCC ?= nvcc
# Override for your installed toolkit and GPU, for example ARCH=sm_90 or sm_100 etc. .
ARCH ?= sm_100
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

all: reduce_demo

reduce_demo: driver.cu reduce.cuh
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) driver.cu -o $@

clean:
	rm -f reduce_demo

.PHONY: all clean
