#include "measurement/cuda.hpp"
#include <algorithm>

namespace measurement {
namespace {
constexpr Size generation_chunk = Size{1} << 20;
constexpr unsigned generation_blocks = 256;
constexpr unsigned generation_threads = 256;
class Generator {
 public:
  Generator() { curand_check(curandCreateGenerator(&value, CURAND_RNG_PSEUDO_PHILOX4_32_10)); }
  ~Generator() { curandDestroyGenerator(value); }
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;
  curandGenerator_t value{};
};
__global__ void map_draws(const double* draws, cuda::std::complex<double>* state, Size offset, Size count) {
  Size id = Size{blockIdx.x} * Size{blockDim.x} + Size{threadIdx.x};
  Size stride = Size{gridDim.x} * Size{blockDim.x};
  for (Size j = id; j < count; j += stride)
    state[offset+j] = cuda::std::complex<double>(draws[2*j], draws[2*j+1]);
}
}
void preflight_device_bytes(Size bytes) {
  std::size_t free_bytes = 0, total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (bytes > free_bytes)
    throw Unsupported("device memory preflight: additional bytes=" + std::to_string(bytes) +
      " free bytes=" + std::to_string(free_bytes) + " total bytes=" + std::to_string(total_bytes));
}
DeviceState generate_state(const Config& c) {
  c.validate();
  Size n = c.state_size();
  allocation_bytes(n, sizeof(cuda::std::complex<double>));
  preflight_device_bytes(checked_add(allocation_bytes(n, sizeof(cuda::std::complex<double>)),
    allocation_bytes(checked_mul(std::min(n, generation_chunk), Size{2}), sizeof(double))));
  DeviceState state(n);
  DeviceBuffer<double> draws(checked_mul(std::min(n, generation_chunk), Size{2}));
  Generator generator;
  curand_check(curandSetPseudoRandomGeneratorSeed(generator.value, c.seed));
  curand_check(curandSetGeneratorOffset(generator.value, 0));
  curand_check(curandSetGeneratorOrdering(generator.value, CURAND_ORDERING_PSEUDO_DEFAULT));
  for (Size offset = 0; offset < n;) {
    Size count = std::min(generation_chunk, n-offset);
    curand_check(curandGenerateUniformDouble(generator.value, draws.data(),
      allocation_bytes(count, 2))); // number of doubles, checked for size_t
    map_draws<<<generation_blocks, generation_threads>>>(draws.data(), state.data(), offset, count);
    cuda_check(cudaGetLastError());
    offset = checked_add(offset, count);
  }
  cuda_check(cudaDeviceSynchronize());
  return state;
}
std::vector<long double> reference_device_state(const Config& c, const DeviceState& state) {
  c.validate();
  if (state.size() != c.state_size()) throw std::invalid_argument("device state size mismatch");
  Size capacity = std::min(c.chunk_elements, state.size());
  allocation_bytes(capacity, sizeof(cuda::std::complex<double>));
  std::vector<cuda::std::complex<double>> staging(static_cast<std::size_t>(capacity));
  std::vector<std::complex<double>> chunk(static_cast<std::size_t>(capacity));
  CpuReference reference(c);
  for (Size offset = 0; offset < state.size();) {
    Size count = std::min(capacity, state.size()-offset);
    cuda_check(cudaMemcpy(staging.data(), state.data()+offset,
      allocation_bytes(count, sizeof(cuda::std::complex<double>)), cudaMemcpyDeviceToHost));
    for (Size i = 0; i < count; ++i) chunk[i] = {staging[i].real(), staging[i].imag()};
    reference.add_chunk(offset, chunk.data(), count);
    offset = checked_add(offset, count);
  }
  return reference.finish();
}
} // namespace measurement
