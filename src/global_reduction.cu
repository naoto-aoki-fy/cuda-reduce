#include "measurement/cuda.hpp"

namespace measurement {
namespace {
// Only 128/256/512 threads are accepted: every shuffle has a full warp.
__device__ double warp_sum(double value) {
  for (unsigned delta = 16; delta; delta /= 2)
    value += __shfl_down_sync(0xffffffffu, value, delta);
  return value;
}
__device__ double block_sum(double value, double* scratch) {
  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  double total = 0;
  if (warp == 0) {
    total = lane < blockDim.x / 32 ? scratch[lane] : 0.0;
    total = warp_sum(total);
  }
  // All scratch readers finish before any thread starts the next bin.
  __syncthreads();
  return total; // Used only by thread zero.
}
__device__ Size accumulator_index(Size g, Size bin, Size threads, Size bins,
                                  bool bin_major) {
  return bin_major ? bin * threads + g : g * bins + bin;
}
__global__ void first_stage(const cuda::std::complex<double>* state, Size count,
                            const unsigned* positions, unsigned k, Size bins,
                            double* sums, double* partials, bool bin_major) {
  __shared__ double scratch[16];
  const Size g = Size{blockIdx.x} * Size{blockDim.x} + Size{threadIdx.x};
  const Size threads = Size{gridDim.x} * Size{blockDim.x};
  for (Size bin = 0; bin < bins; ++bin)
    sums[accumulator_index(g, bin, threads, bins, bin_major)] = 0.0;
  for (Size j = g; j < count; j += threads) {
    const auto amplitude = state[j];
    const double re = amplitude.real(), im = amplitude.imag();
    Size bin = 0;
    for (unsigned t = 0; t < k; ++t)
      bin |= ((j >> positions[t]) & Size{1}) << t;
    sums[accumulator_index(g, bin, threads, bins, bin_major)] += re * re + im * im;
  }
  for (Size bin = 0; bin < bins; ++bin) {
    const double total = block_sum(
      sums[accumulator_index(g, bin, threads, bins, bin_major)], scratch);
    // Fixed bin-major partials, independent of the accumulator layout.
    if (threadIdx.x == 0) partials[bin * Size{gridDim.x} + Size{blockIdx.x}] = total;
  }
}
__global__ void second_stage(const double* partials, double* output, Size bins) {
  __shared__ double scratch[16];
  const Size blocks = Size{gridDim.x};
  for (Size bin = Size{blockIdx.x}; bin < bins; bin += blocks) {
    double value = 0;
    for (Size block = Size{threadIdx.x}; block < blocks; block += Size{blockDim.x})
      value += partials[bin * blocks + block];
    const double total = block_sum(value, scratch);
    if (threadIdx.x == 0) output[bin] = total;
  }
}
Size checked_blocks(const Config& c) {
  c.validate();
  int device;
  cuda_check(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device));
  const Size blocks = launch_blocks(c, static_cast<Size>(prop.multiProcessorCount));
  if (blocks > static_cast<Size>(prop.maxGridSize[0]) ||
      c.block_size > static_cast<unsigned>(prop.maxThreadsPerBlock) ||
      c.block_size > static_cast<unsigned>(prop.maxThreadsDim[0]))
    throw Unsupported("requested grid or block exceeds device limits");
  cudaFuncAttributes first{}, second{};
  cuda_check(cudaFuncGetAttributes(&first, first_stage));
  cuda_check(cudaFuncGetAttributes(&second, second_stage));
  if (c.block_size > static_cast<unsigned>(first.maxThreadsPerBlock) ||
      c.block_size > static_cast<unsigned>(second.maxThreadsPerBlock) ||
      first.sharedSizeBytes > prop.sharedMemPerBlock ||
      second.sharedSizeBytes > prop.sharedMemPerBlock)
    throw Unsupported("requested block exceeds reduction kernel resource limits");
  // Check every product before allocating any workspace.
  Size bytes = allocation_bytes(workspace_elements(blocks, Size{c.block_size}, c.bins()), sizeof(double));
  bytes = checked_add(bytes, allocation_bytes(checked_mul(blocks, c.bins()), sizeof(double)));
  bytes = checked_add(bytes, allocation_bytes(c.bins(), sizeof(double)));
  checked_add(bytes, allocation_bytes(c.positions.size(), sizeof(unsigned)));
  return blocks;
}
class Event {
 public:
  Event() { cuda_check(cudaEventCreate(&event)); }
  ~Event() { cudaEventDestroy(event); }
  cudaEvent_t event{};
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};
}
const char* layout_name(GlobalLayout layout) {
  switch (layout) {
    case GlobalLayout::thread_major: return "global thread-major";
    case GlobalLayout::bin_major: return "global bin-major";
  }
  throw std::invalid_argument("unknown global accumulator layout");
}
GlobalReduction::GlobalReduction(const Config& c, GlobalLayout layout)
  : config_(c), layout_(layout), blocks_(checked_blocks(c)),
    threads_(checked_mul(blocks_, Size{c.block_size})), positions_(c.positions.size()),
    sums_(checked_mul(threads_, c.bins())), partials_(checked_mul(blocks_, c.bins())),
    output_(c.bins()) {
  layout_name(layout); // Reject invalid enum values rather than choosing a fallback.
  cuda_check(cudaMemcpy(positions_.data(), c.positions.data(),
    allocation_bytes(c.positions.size(), sizeof(unsigned)), cudaMemcpyHostToDevice));
}
void GlobalReduction::execute(const DeviceState& state, cudaStream_t stream) {
  if (state.size() != config_.state_size()) throw std::invalid_argument("device state size mismatch");
  const auto grid = static_cast<unsigned>(blocks_);
  first_stage<<<grid, config_.block_size, 0, stream>>>(state.data(), state.size(),
    positions_.data(), static_cast<unsigned>(config_.positions.size()), config_.bins(),
    sums_.data(), partials_.data(), layout_ == GlobalLayout::bin_major);
  cuda_check(cudaGetLastError());
  second_stage<<<grid, config_.block_size, 0, stream>>>(partials_.data(), output_.data(), config_.bins());
  cuda_check(cudaGetLastError());
}
void GlobalReduction::poison(cudaStream_t stream) {
  for (auto* buffer : {&sums_, &partials_, &output_})
    cuda_check(cudaMemsetAsync(buffer->data(), 0xff,
      allocation_bytes(buffer->size(), sizeof(double)), stream));
}
Result GlobalReduction::run(const DeviceState& state, std::vector<double>& output,
                             cudaStream_t stream) {
  Result result;
  output.clear();
  try {
    // Events, output host allocation, and workspaces are outside the timed region.
    Event start, stop;
    output.resize(static_cast<std::size_t>(config_.bins()));
    for (Size i = 0; i < config_.warmups; ++i) execute(state, stream);
    cuda_check(cudaEventRecord(start.event, stream));
    for (Size i = 0; i < config_.repetitions; ++i) execute(state, stream);
    cuda_check(cudaEventRecord(stop.event, stream));
    cuda_check(cudaEventSynchronize(stop.event));
    float elapsed;
    cuda_check(cudaEventElapsedTime(&elapsed, start.event, stop.event));
    result.elapsed_ms = double(elapsed) / double(config_.repetitions);
    cuda_check(cudaMemcpyAsync(output.data(), output_.data(),
      allocation_bytes(output_.size(), sizeof(double)), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    result.status = Status::success;
  } catch (const Unsupported& e) {
    result.status = Status::unsupported;
    result.reason = e.what();
  } catch (const std::exception& e) {
    result.status = Status::failure;
    result.reason = e.what();
  }
  if (result.status != Status::success) { output.clear(); result.elapsed_ms.reset(); }
  return result;
}
} // namespace measurement
