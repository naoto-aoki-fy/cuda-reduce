.DELETE_ON_ERROR:

CXX ?= g++
NVCC ?= nvcc
AR ?= ar
CPPFLAGS += -Iinclude -I$(BUILD)
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
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/libmeasurement_core.a: $(HOST_OBJECT)
	$(AR) rcs $@ $^

$(BUILD)/cuda.o: src/cuda.cu include/measurement/cuda.hpp include/measurement/gather.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/global_reduction.o: src/global_reduction.cu include/measurement/cuda.hpp include/measurement/gather.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(BUILD)/cub_reduction.o: src/cub_reduction.cu include/measurement/gather.hpp include/measurement/cuda.hpp include/measurement/core.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -std=c++17 -MMD -MP -c $< -o $@

$(CUDA_OBJECT): $(BUILD)/build_info.hpp

$(BUILD)/main.o: src/main.cu $(BUILD)/build_info.hpp include/measurement/cuda.hpp include/measurement/gather.hpp include/measurement/core.hpp | $(BUILD)
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

# Recheck metadata each invocation; only rewrite the header if its content changes.
.PHONY: FORCE benchmark-smoke benchmark-full benchmark-qubits-9-16 profile
$(BUILD)/build_info.hpp: export MEASUREMENT_NVCC = $(NVCC)
$(BUILD)/build_info.hpp: export MEASUREMENT_NVCCFLAGS = $(NVCCFLAGS)
$(BUILD)/build_info.hpp: export MEASUREMENT_CPPFLAGS = $(CPPFLAGS)
$(BUILD)/build_info.hpp: FORCE | $(BUILD)
	python3 tools/build_info.py $@

BENCHMARK_ARGS ?=
benchmark-smoke: cuda
	./$(BUILD)/measurement_validate --preset smoke --results $(BUILD)/smoke $(BENCHMARK_ARGS)
benchmark-full: cuda
	./$(BUILD)/measurement_validate --preset full --results $(BUILD)/full $(BENCHMARK_ARGS)
benchmark-qubits-9-16: cuda
	@set -e; for qubits in 9 10 11 12 13 14 15 16; do \
		./$(BUILD)/measurement_validate $(BENCHMARK_ARGS) --preset smoke --qubits $$qubits --results $(BUILD)/qubits-$$qubits; \
	done
profile: cuda
	python3 tools/profile_benchmark.py --binary $(BUILD)/measurement_validate $(PROFILE_ARGS)
