#pragma once
#include "measurement/core.hpp"
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <curand.h>
#include <stdexcept>
#include <utility>

namespace measurement {
inline void cuda_check(cudaError_t code) {
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
} // namespace measurement
