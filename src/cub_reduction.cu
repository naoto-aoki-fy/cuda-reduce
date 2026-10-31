#include "measurement/cuda.hpp"
#include "measurement/gather.hpp"
#include <cub/version.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <sstream>
#include <algorithm>

namespace measurement {
std::string cub_version_description() {
  std::ostringstream out;
  out << "CUDA headers=" << CUDART_VERSION << " CUB=" << CUB_VERSION
      << " (CCCL release source audit: 2.8.2); BlockReduce=BLOCK_REDUCE_WARP_REDUCTIONS"
      << "; DeviceSegmentedReduce::Sum=one CTA per segment, policy-hub warp reductions";
  return out.str();
}
void require_audited_cub() {
  // Unknown releases must be inspected before their executed library path is enabled.
  if (CUB_VERSION != 200802)
    throw Unsupported("CUB API/64-bit-offset/atomic-policy source audit covers only CUB 2.8.2; "
      "selected CUB_VERSION=" + std::to_string(CUB_VERSION) + "; see docs/cub-baselines.md");
}
namespace {
struct GatherNorm {
  const cuda::std::complex<double>* state = nullptr;
  GatherMap map;
  __host__ __device__ double operator()(LogicalOffset v) const {
    const auto a = state[map(v)];
    const double re = a.real(), im = a.imag();
    return re * re + im * im;
  }
};
using InputIterator = LogicalIterator<GatherNorm, double>;
static_assert(sizeof(std::iterator_traits<InputIterator>::difference_type) == 8);
static_assert(sizeof(std::iterator_traits<OffsetIterator>::value_type) == 8);
Config supported_config(const Config& c) {
  c.validate();
  require_audited_cub();
  // q=63's exclusive endpoint cannot be represented by a signed difference_type.
  if (c.state_size() > Size(std::numeric_limits<LogicalOffset>::max()))
    throw Unsupported("DeviceSegmentedReduce exclusive N exceeds signed 64-bit iterator range");
  return c;
}
cudaError_t sum(const Config& c, const GatherMap& map, const cuda::std::complex<double>* state, double* output,
                void* temporary, std::size_t& bytes, cudaStream_t stream) {
  InputIterator input{{state, map}, 0};
  OffsetIterator offsets{{static_cast<LogicalOffset>(map.length)}, 0};
  // B <= 256; only the number of segments is int. Offset values and input
  // iterator differences are int64_t through the entire audited Sum call path.
  return cub::DeviceSegmentedReduce::Sum(temporary, bytes, input, output,
    static_cast<int>(c.bins()), offsets, offsets + 1, stream);
}
struct Event {
  cudaEvent_t value{};
  Event() { cuda_check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event&) = delete;
};
}
SegmentedReduction::SegmentedReduction(const Config& c)
  : config_(supported_config(c)), map_(config_), output_(config_.bins()) {
  // Query performs no input reads; state is supplied only during execution.
  cuda_check(sum(config_, map_, nullptr, output_.data(), nullptr, temporary_bytes_, nullptr));
  temporary_ = std::make_unique<DeviceBuffer<unsigned char>>(std::max(std::size_t{1}, temporary_bytes_));
}
void SegmentedReduction::execute(const DeviceState& state, cudaStream_t stream) {
  if (state.size() != config_.state_size()) throw std::invalid_argument("device state size mismatch");
  cuda_check(sum(config_, map_, state.data(), output_.data(), temporary_->data(), temporary_bytes_, stream));
}
Result SegmentedReduction::run(const DeviceState& state, std::vector<double>& output, cudaStream_t stream) {
  Result result;
  output.clear();
  try {
    Event start, stop;
    output.resize(config_.bins());
    for (Size i = 0; i < config_.warmups; ++i) execute(state, stream);
    cuda_check(cudaEventRecord(start.value, stream));
    for (Size i = 0; i < config_.repetitions; ++i) execute(state, stream);
    cuda_check(cudaEventRecord(stop.value, stream));
    cuda_check(cudaEventSynchronize(stop.value));
    float elapsed;
    cuda_check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
    result.elapsed_ms = double(elapsed) / double(config_.repetitions);
    cuda_check(cudaMemcpyAsync(output.data(), output_.data(), allocation_bytes(output_.size(), sizeof(double)),
      cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    result.status = Status::success;
  } catch (const Unsupported& e) {
    result.status = Status::unsupported; result.reason = e.what();
  } catch (const std::exception& e) {
    result.status = Status::failure; result.reason = e.what();
  }
  if (result.status != Status::success) { output.clear(); result.elapsed_ms.reset(); }
  return result;
}
} // namespace measurement
