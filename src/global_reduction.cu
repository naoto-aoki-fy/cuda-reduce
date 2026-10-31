#include "measurement/cuda.hpp"
#include <sstream>
#include <climits>
#include <algorithm>

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
// Shared accumulators and reduction scratch are separate, double-aligned regions.
__global__ void shared_first_stage(const cuda::std::complex<double>* state, Size count,
    const unsigned* positions, unsigned k, Size bins, double* partials, bool bin_major) {
  extern __shared__ double sums[];
  __shared__ double scratch[16];
  const Size t = threadIdx.x, size = blockDim.x;
  const Size g = Size{blockIdx.x} * size + t;
  const Size threads = Size{gridDim.x} * size;
  for (Size bin = 0; bin < bins; ++bin)
    sums[accumulator_index(t, bin, size, bins, bin_major)] = 0;
  for (Size j = g; j < count; j += threads) {
    const auto amplitude = state[j];
    Size bin = 0;
    for (unsigned bit = 0; bit < k; ++bit)
      bin |= ((j >> positions[bit]) & Size{1}) << bit;
    const double re = amplitude.real(), im = amplitude.imag();
    sums[accumulator_index(t, bin, size, bins, bin_major)] += re * re + im * im;
  }
  for (Size bin = 0; bin < bins; ++bin) {
    const double total = block_sum(sums[accumulator_index(t, bin, size, bins, bin_major)], scratch);
    if (t == 0) partials[bin * Size{gridDim.x} + Size{blockIdx.x}] = total;
  }
}
// Fixed capacity of eight doubles; no dynamically indexed automatic array.
// k and positions remain runtime inputs. Residency still requires binary evidence.
__global__ void register_first_stage(const cuda::std::complex<double>* state, Size count,
    const unsigned* positions, unsigned k, Size bins, double* partials) {
  __shared__ double scratch[16];
  double a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
  const Size g = Size{blockIdx.x} * Size{blockDim.x} + Size{threadIdx.x};
  const Size threads = Size{gridDim.x} * Size{blockDim.x};
  for (Size j = g; j < count; j += threads) {
    const auto amplitude = state[j];
    const double re = amplitude.real(), im = amplitude.imag();
    const double value = re * re + im * im;
    Size bin = 0;
    for (unsigned bit = 0; bit < k; ++bit)
      bin |= ((j >> positions[bit]) & Size{1}) << bit;
    switch (bin) {
      case 0: a0 += value; break; case 1: a1 += value; break;
      case 2: a2 += value; break; case 3: a3 += value; break;
      case 4: a4 += value; break; case 5: a5 += value; break;
      case 6: a6 += value; break; case 7: a7 += value; break;
    }
  }
  for (Size bin = 0; bin < bins; ++bin) {
    double value = 0;
    switch (bin) {
      case 0: value=a0; break; case 1: value=a1; break;
      case 2: value=a2; break; case 3: value=a3; break;
      case 4: value=a4; break; case 5: value=a5; break;
      case 6: value=a6; break; case 7: value=a7; break;
    }
    const double total = block_sum(value, scratch);
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
Size checked_blocks(const Config& c, AccumulatorStorage storage) {
  c.validate();
  if (storage == AccumulatorStorage::registers && c.bins() > 8)
    throw Unsupported("register capacity=8 doubles; supported runtime k=1..3; layout=n/a");
  const auto resources = reduction_resources(c, storage);
  if (storage == AccumulatorStorage::registers) {
    throw Unsupported("register capacity=8 doubles; runtime k=1..3; layout=n/a; residency unverified: "
                      "no compiler/disassembly audit for this build; " + resource_description(resources));
  }
  const Size blocks = resources.blocks;
  Size bytes = storage == AccumulatorStorage::global
    ? allocation_bytes(workspace_elements(blocks, Size{c.block_size}, c.bins()), sizeof(double)) : 0;
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
const char* storage_name(AccumulatorStorage storage) {
  switch (storage) {
    case AccumulatorStorage::global: return "global";
    case AccumulatorStorage::shared: return "shared";
    case AccumulatorStorage::registers: return "registers (layout=n/a, capacity=8 doubles, runtime k=1..3)";
  }
  throw std::invalid_argument("unknown accumulator storage");
}
std::string resource_description(const ReductionResources& r) {
  std::ostringstream out;
  out << "shared limits default=" << r.default_shared_limit << " optin=" << r.optin_shared_limit
      << " kernel-dynamic=" << r.dynamic_shared_limit;
  unsigned stage = 0;
  for (const auto& k : {r.first, r.second})
    out << "; stage=" << ++stage << " registers/thread=" << k.registers_per_thread
        << " local/thread=" << k.local_bytes_per_thread << " static-shared=" << k.static_shared_bytes
        << " requested-dynamic=" << k.dynamic_shared_bytes << " max-threads=" << k.max_threads
        << " active-blocks/SM=" << k.active_blocks_per_sm << " occupancy=" << k.theoretical_occupancy;
  return out.str();
}
ReductionResources reduction_resources(const Config& c, AccumulatorStorage storage) {
  c.validate();
  storage_name(storage);
  int device;
  cuda_check(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device));
  ReductionResources r;
  r.blocks = launch_blocks(c, static_cast<Size>(prop.multiProcessorCount));
  r.default_shared_limit = prop.sharedMemPerBlock;
  r.optin_shared_limit = prop.sharedMemPerBlockOptin;
  if (r.blocks > static_cast<Size>(prop.maxGridSize[0]) ||
      c.block_size > static_cast<unsigned>(prop.maxThreadsPerBlock) ||
      c.block_size > static_cast<unsigned>(prop.maxThreadsDim[0]))
    throw Unsupported("requested grid or block exceeds device limits");
  const void* first = storage == AccumulatorStorage::global ? reinterpret_cast<const void*>(first_stage)
    : storage == AccumulatorStorage::shared ? reinterpret_cast<const void*>(shared_first_stage)
    : reinterpret_cast<const void*>(register_first_stage);
  const void* second = reinterpret_cast<const void*>(second_stage);
  cudaFuncAttributes fa{}, sa{};
  cuda_check(cudaFuncGetAttributes(&fa, first));
  cuda_check(cudaFuncGetAttributes(&sa, second));
  // 8*S*B is already double aligned. Scratch is static; no padding is needed
  // between separately allocated static and extern-double shared regions.
  const Size dynamic = storage == AccumulatorStorage::shared
    ? allocation_bytes(checked_mul(Size{c.block_size}, c.bins()), sizeof(double)) : 0;
  auto fill = [&](KernelResources& k, const cudaFuncAttributes& a, Size bytes) {
    k.registers_per_thread = a.numRegs; k.local_bytes_per_thread = a.localSizeBytes;
    k.static_shared_bytes = a.sharedSizeBytes; k.dynamic_shared_bytes = bytes;
    k.max_threads = a.maxThreadsPerBlock;
  };
  fill(r.first, fa, dynamic); fill(r.second, sa, 0);
  r.dynamic_shared_limit = fa.maxDynamicSharedSizeBytes;
  const Size total = checked_add(dynamic, fa.sharedSizeBytes);
  if (dynamic > static_cast<Size>(fa.maxDynamicSharedSizeBytes) &&
      total > r.default_shared_limit && dynamic <= INT_MAX && total <= r.optin_shared_limit) {
    const auto status = cudaFuncSetAttribute(first, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(dynamic));
    if (status != cudaSuccess)
      throw Unsupported(std::string("dynamic shared opt-in failed: ") + cudaGetErrorString(status)
                        + "; " + resource_description(r));
    cuda_check(cudaFuncGetAttributes(&fa, first));
    r.dynamic_shared_limit = fa.maxDynamicSharedSizeBytes;
  }
  if (total > std::max(r.default_shared_limit, r.optin_shared_limit) ||
      dynamic > static_cast<Size>(fa.maxDynamicSharedSizeBytes) ||
      sa.sharedSizeBytes > r.default_shared_limit ||
      c.block_size > static_cast<unsigned>(fa.maxThreadsPerBlock) ||
      c.block_size > static_cast<unsigned>(sa.maxThreadsPerBlock))
    throw Unsupported("kernel resource limit: " + resource_description(r));
  auto occupancy = [&](const void* kernel, KernelResources& k) {
    const auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &k.active_blocks_per_sm, kernel, c.block_size, k.dynamic_shared_bytes);
    if (status != cudaSuccess)
      throw Unsupported(std::string("occupancy query: ") + cudaGetErrorString(status) + "; " + resource_description(r));
    k.theoretical_occupancy = double(k.active_blocks_per_sm) * c.block_size / prop.maxThreadsPerMultiProcessor;
    if (!k.active_blocks_per_sm) throw Unsupported("zero kernel residency: " + resource_description(r));
  };
  occupancy(first, r.first); occupancy(second, r.second);
  return r;
}
const char* layout_name(GlobalLayout layout) {
  switch (layout) {
    case GlobalLayout::thread_major: return "global thread-major";
    case GlobalLayout::bin_major: return "global bin-major";
  }
  throw std::invalid_argument("unknown global accumulator layout");
}
GlobalReduction::GlobalReduction(const Config& c, GlobalLayout layout, AccumulatorStorage storage)
  : config_(c), layout_(layout), storage_(storage), blocks_(checked_blocks(c, storage)),
    threads_(checked_mul(blocks_, Size{c.block_size})), positions_(c.positions.size()),
    sums_(storage == AccumulatorStorage::global ? checked_mul(threads_, c.bins()) : 0), partials_(checked_mul(blocks_, c.bins())),
    output_(c.bins()) {
  dynamic_shared_ = storage == AccumulatorStorage::shared
    ? allocation_bytes(checked_mul(Size{c.block_size}, c.bins()), sizeof(double)) : 0;
  layout_name(layout); // Reject invalid enum values rather than choosing a fallback.
  cuda_check(cudaMemcpy(positions_.data(), c.positions.data(),
    allocation_bytes(c.positions.size(), sizeof(unsigned)), cudaMemcpyHostToDevice));
}
void GlobalReduction::execute(const DeviceState& state, cudaStream_t stream) {
  if (state.size() != config_.state_size()) throw std::invalid_argument("device state size mismatch");
  const auto grid = static_cast<unsigned>(blocks_);
  if (storage_ == AccumulatorStorage::shared)
    shared_first_stage<<<grid, config_.block_size, dynamic_shared_, stream>>>(state.data(), state.size(),
      positions_.data(), static_cast<unsigned>(config_.positions.size()), config_.bins(),
      partials_.data(), layout_ == GlobalLayout::bin_major);
  else if (storage_ == AccumulatorStorage::registers)
    register_first_stage<<<grid, config_.block_size, 0, stream>>>(state.data(), state.size(),
      positions_.data(), static_cast<unsigned>(config_.positions.size()), config_.bins(), partials_.data());
  else
    first_stage<<<grid, config_.block_size, 0, stream>>>(state.data(), state.size(),
      positions_.data(), static_cast<unsigned>(config_.positions.size()), config_.bins(),
      sums_.data(), partials_.data(), layout_ == GlobalLayout::bin_major);
  cuda_check(cudaGetLastError());
  second_stage<<<grid, config_.block_size, 0, stream>>>(partials_.data(), output_.data(), config_.bins());
  cuda_check(cudaGetLastError());
}
void GlobalReduction::poison(cudaStream_t stream) {
  for (auto* buffer : {&sums_, &partials_, &output_})
    if (buffer->size()) cuda_check(cudaMemsetAsync(buffer->data(), 0xff,
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
