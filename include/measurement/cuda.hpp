#pragma once
#include "measurement/core.hpp"
#include "measurement/gather.hpp"
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <curand.h>
#include <stdexcept>
#include <utility>
#include <memory>

namespace measurement {
class Unsupported : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
inline void cuda_check(cudaError_t code) {
  if (code == cudaErrorMemoryAllocation)
    throw Unsupported(std::string("CUDA allocation: ") + cudaGetErrorString(code));
  if (code == cudaErrorLaunchOutOfResources)
    throw Unsupported(std::string("CUDA launch resources: ") + cudaGetErrorString(code));
  if (code != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(code));
}
inline void curand_check(curandStatus_t code) {
  if (code != CURAND_STATUS_SUCCESS) throw std::runtime_error("cuRAND status " + std::to_string(code));
}
template<class T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(Size count) : count_(count) {
    if (count) cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), allocation_bytes(count, sizeof(T))));
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
  T* data() const { return data_; }
  Size size() const { return count_; }
 private:
  T* data_ = nullptr;
  Size count_;
};
using DeviceState = DeviceBuffer<cuda::std::complex<double>>;
void preflight_device_bytes(Size bytes);
Size segmented_workspace_bytes(const Config&);
DeviceState generate_state(const Config& config);
std::vector<long double> reference_device_state(const Config& config, const DeviceState& state);
const char* layout_name(GlobalLayout layout);
const char* storage_name(AccumulatorStorage storage);
struct KernelResources {
  int registers_per_thread = 0;
  int binary_version = 0, ptx_version = 0;
  Size local_bytes_per_thread = 0, static_shared_bytes = 0, dynamic_shared_bytes = 0;
  int max_threads = 0, active_blocks_per_sm = 0;
  double theoretical_occupancy = 0;
};
std::string cub_version_description();
struct ReductionResources {
  Size sm_count = 0;
  Size blocks = 0;
  KernelResources first, second;
  Size default_shared_limit = 0, optin_shared_limit = 0;
  int dynamic_shared_limit = 0;
  bool register_residency_verified = false;
};
// Resource inspection does not certify atomic policy or register residency.
ReductionResources reduction_resources(const Config&, AccumulatorStorage,
  BlockPrimitive = BlockPrimitive::custom);
std::string resource_description(const ReductionResources&);

// Owns reusable buffers; construction and position upload are outside timing.
// Use on the device where it was constructed. Calls on one workspace are serial.
class GlobalReduction {
 public:
  GlobalReduction(const Config& config, GlobalLayout layout,
                  AccumulatorStorage storage = AccumulatorStorage::global,
                  BlockPrimitive primitive = BlockPrimitive::custom);
  void execute(const DeviceState& state, cudaStream_t stream = nullptr);
  Result run(const DeviceState& state, std::vector<double>& output,
             cudaStream_t stream = nullptr);
  // Validation helper: poison all writable buffers before checking overwrite.
  void poison(cudaStream_t stream = nullptr);
  Size blocks() const { return blocks_; }
  Size threads() const { return threads_; }
 private:
  Config config_;
  GlobalLayout layout_;
  AccumulatorStorage storage_;
  BlockPrimitive primitive_;
  Size dynamic_shared_ = 0;
  Size blocks_, threads_;
  DeviceBuffer<unsigned> positions_;
  DeviceBuffer<double> sums_, partials_, output_;
};
// CUB chooses its launch configuration; custom tuning fields are not applicable.
class SegmentedReduction {
 public:
  explicit SegmentedReduction(const Config&);
  void execute(const DeviceState&, cudaStream_t = nullptr);
  Result run(const DeviceState&, std::vector<double>&, cudaStream_t = nullptr);
 private:
  Config config_;
  GatherMap map_;
  DeviceBuffer<double> output_;
  std::size_t temporary_bytes_ = 0;
  std::unique_ptr<DeviceBuffer<unsigned char>> temporary_;
};
} // namespace measurement
