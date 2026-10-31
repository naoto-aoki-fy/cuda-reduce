.DELETE_ON_ERROR:

CXX ?= g++
NVCC ?= nvcc
AR ?= ar
CPPFLAGS += -Iinclude
CXXFLAGS ?= -O2 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O2
BUILD := build
HOST_OBJECT := $(BUILD)/core.o
CUDA_OBJECT := $(BUILD)/cuda.o $(BUILD)/global_reduction.o $(BUILD)/cub_reduction.o
CUDA_AVAILABLE := $(shell command -v $(firstword $(NVCC)) 2>/dev/null)

.PHONY: all host cuda test test-host test-cuda test-cub sanitize clean
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

$(BUILD)/cuda.o: src/cuda.cu include/measurement/cuda.hpp include/measurement/gather.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/global_reduction.o: src/global_reduction.cu include/measurement/cuda.hpp include/measurement/gather.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/cub_reduction.o: src/cub_reduction.cu include/measurement/gather.hpp include/measurement/cuda.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/main.o: src/main.cu include/measurement/cuda.hpp include/measurement/gather.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/measurement_validate: $(BUILD)/main.o $(CUDA_OBJECT) $(BUILD)/libmeasurement_core.a
	$(NVCC) $(NVCCFLAGS) -std=c++17 $^ -lcurand -o $@

$(BUILD)/test_core: tests/core.cpp $(BUILD)/libmeasurement_core.a include/measurement/gather.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $< $(BUILD)/libmeasurement_core.a -o $@

$(BUILD)/test_reduction: tests/reduction.cu $(CUDA_OBJECT) $(BUILD)/libmeasurement_core.a
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 $^ -lcurand -o $@

$(BUILD)/test_cub: tests/cub.cu $(CUDA_OBJECT) $(BUILD)/libmeasurement_core.a
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 $^ -lcurand -o $@

test: test-host
test-host: $(BUILD)/test_core
	./$(BUILD)/test_core
test-cuda: $(BUILD)/test_reduction $(BUILD)/test_cub
	./$(BUILD)/test_reduction
	./$(BUILD)/test_cub
test-cub: $(BUILD)/test_cub
	./$(BUILD)/test_cub
sanitize: $(BUILD)/test_reduction
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(BUILD)/test_reduction --quick
	compute-sanitizer --tool synccheck --error-exitcode 1 ./$(BUILD)/test_reduction --quick
	compute-sanitizer --tool racecheck --error-exitcode 1 ./$(BUILD)/test_reduction --quick

clean:
	$(RM) -r $(BUILD)

-include $(BUILD)/*.d

# Explicit architecture is required; artifacts never enable register timing.
.PHONY: cuda-evidence
cuda-evidence:
	python3 tools/capture_cuda_evidence.py --arch '$(CUDA_ARCH)' --nvcc '$(NVCC)' --flags='$(NVCCFLAGS)'
