.DELETE_ON_ERROR:

CXX ?= g++
NVCC ?= nvcc
AR ?= ar
CPPFLAGS += -Iinclude
CXXFLAGS ?= -O2 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O2
BUILD := build
HOST_OBJECT := $(BUILD)/core.o
CUDA_OBJECT := $(BUILD)/cuda.o
CUDA_AVAILABLE := $(shell command -v $(firstword $(NVCC)) 2>/dev/null)

.PHONY: all host cuda clean
all: host
ifneq ($(CUDA_AVAILABLE),)
all: cuda
endif

host: $(BUILD)/libmeasurement_core.a
cuda: $(BUILD)/measurement_validate

$(BUILD):
	mkdir -p $@

$(HOST_OBJECT): src/core.cpp include/measurement/core.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/libmeasurement_core.a: $(HOST_OBJECT)
	$(AR) rcs $@ $^

$(CUDA_OBJECT): src/cuda.cu include/measurement/cuda.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/main.o: src/main.cu include/measurement/cuda.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/measurement_validate: $(BUILD)/main.o $(CUDA_OBJECT) $(BUILD)/libmeasurement_core.a
	$(NVCC) $(NVCCFLAGS) -std=c++17 $^ -lcurand -o $@

clean:
	$(RM) -r $(BUILD)

-include $(BUILD)/*.d
