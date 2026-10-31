#pragma once
#include "measurement/core.hpp"
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <curand.h>
#include <stdexcept>
#include <utility>

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
DeviceState generate_state(const Config& config);
std::vector<long double> reference_device_state(const Config& config, const DeviceState& state);
enum class GlobalLayout { thread_major, bin_major };
const char* layout_name(GlobalLayout layout);

// Owns reusable buffers; construction and position upload are outside timing.
// Use on the device where it was constructed. Calls on one workspace are serial.
class GlobalReduction {
 public:
  GlobalReduction(const Config& config, GlobalLayout layout);
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
  Size blocks_, threads_;
  DeviceBuffer<unsigned> positions_;
  DeviceBuffer<double> sums_, partials_, output_;
};
} // namespace measurement
