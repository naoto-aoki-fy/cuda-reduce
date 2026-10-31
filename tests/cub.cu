#include "measurement/cuda.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace measurement;
namespace {
void compare(const std::vector<double>& actual, const std::vector<long double>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("CUB output size mismatch");
  for (std::size_t b = 0; b < actual.size(); ++b)
    if (!std::isfinite(actual[b]) ||
        std::abs(static_cast<long double>(actual[b]) - expected[b]) > 1e-12L + 1e-11L * std::abs(expected[b]))
      throw std::runtime_error("CUB CPU mismatch at bin " + std::to_string(b));
}
template<class Reduction>
void reuse(Reduction& reduction, DeviceState& state,
           const std::vector<cuda::std::complex<double>>& values,
           const std::vector<long double>& expected, cudaStream_t stream) {
  for (unsigned round = 0; round < 3; ++round) {
    if (round == 2) cuda_check(cudaMemsetAsync(state.data(), 0,
      allocation_bytes(state.size(), sizeof(values[0])), stream));
    else cuda_check(cudaMemcpyAsync(state.data(), values.data(),
      allocation_bytes(state.size(), sizeof(values[0])), cudaMemcpyHostToDevice, stream));
    std::vector<double> output;
    auto result = reduction.run(state, output, stream);
    if (result.status != Status::success || !result.elapsed_ms)
      throw std::runtime_error(result.reason);
    compare(output, round == 2 ? std::vector<long double>(expected.size(), 0) : expected);
  }
}
struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
void check(Config c) {
  c.warmups = 1; c.repetitions = 2; c.sm_multiplier = 1;
  std::vector<std::complex<double>> host(c.state_size());
  std::vector<cuda::std::complex<double>> values;
  for (Size j = 0; j < c.state_size(); ++j) {
    host[j] = {double(int(j % 19) - 9) / 7, double(int(j % 23) - 11) / 13};
    values.emplace_back(host[j].real(), host[j].imag());
  }
  CpuReference cpu(c); cpu.add_chunk(0, host.data(), host.size());
  const auto expected = cpu.finish();
  DeviceState state(c.state_size());
  Stream stream;
  // Segmented runs exactly once per input/position set, independent of S/layout.
  SegmentedReduction segmented(c);
  reuse(segmented, state, values, expected, stream.value);
  for (unsigned block : {128u, 256u, 512u}) {
    c.block_size = block;
    for (auto storage : {AccumulatorStorage::global, AccumulatorStorage::shared})
    for (auto layout : {GlobalLayout::thread_major, GlobalLayout::bin_major}) {
      std::unique_ptr<GlobalReduction> reduction;
      try {
        reduction = std::make_unique<GlobalReduction>(c, layout, storage,
          ReductionMethod::two_kernel, BlockPrimitive::cub);
      } catch (const Unsupported& e) {
        std::cout << "CUB BlockReduce S=" << block << " storage=" << storage_name(storage)
                  << " layout=" << layout_name(layout) << " k=" << c.positions.size()
                  << " unsupported: " << e.what() << '\n';
        continue;
      }
      reduction->poison(stream.value);
      reuse(*reduction, state, values, expected, stream.value);
    }
  }
}
}
int main() {
  try {
    require_audited_cub();
    std::cout << cub_version_description() << '\n';
    for (unsigned q : {3u, 8u}) for (unsigned k = 1; k <= q; ++k) {
      std::vector<std::vector<unsigned>> positions;
      for (const auto& band : {"low", "middle", "high"})
        positions.push_back(contiguous_positions(q, k, band));
      std::vector<unsigned> scattered;
      for (unsigned t = 0; t < k; ++t) scattered.push_back(t % 2 ? t/2 : q - 1 - t/2);
      positions.push_back(scattered);
      for (auto p : positions) for (bool reversed : {false, true}) {
        if (reversed) std::reverse(p.begin(), p.end());
        Config c; c.qubits = q; c.positions = p; check(c);
      }
    }
    for (auto method : {ReductionMethod::two_kernel, ReductionMethod::cooperative})
    for (unsigned k : {1u, 3u, 4u, 8u}) {
      Config c; c.qubits = 8; c.positions = contiguous_positions(8, k, "low");
      try {
        GlobalReduction rejected(c, GlobalLayout::thread_major, AccumulatorStorage::registers,
          method, BlockPrimitive::cub);
        throw std::runtime_error("CUB register evidence gate unexpectedly opened");
      } catch (const Unsupported& e) { std::cout << "CUB registers unsupported: " << e.what() << '\n'; }
    }
    for (auto storage : {AccumulatorStorage::global, AccumulatorStorage::shared}) {
      Config c;
      try {
        GlobalReduction rejected(c, GlobalLayout::thread_major, storage,
          ReductionMethod::cooperative, BlockPrimitive::cub);
        throw std::runtime_error("CUB cooperative policy gate unexpectedly opened");
      } catch (const Unsupported& e) { std::cout << "CUB cooperative unsupported: " << e.what() << '\n'; }
    }
    std::cout << "CUB baseline checks passed for executable configurations\n";
  } catch (const Unsupported& e) {
    std::cerr << "unsupported: " << e.what() << '\n'; return 77;
  } catch (const std::exception& e) {
    std::cerr << "failure: " << e.what() << '\n'; return 1;
  }
}
