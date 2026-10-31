#include "measurement/cuda.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace measurement;
namespace {
struct Stream {
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  cudaStream_t value{};
};
void compare(const std::vector<double>& actual, const std::vector<long double>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("output size mismatch");
  for (std::size_t b = 0; b < actual.size(); ++b)
    if (!std::isfinite(actual[b]) ||
        std::abs(static_cast<long double>(actual[b]) - expected[b]) > 1e-12L + 1e-11L * std::abs(expected[b]))
      throw std::runtime_error("reference mismatch at bin " + std::to_string(b));
}
void check_case(Config c, const std::vector<std::complex<double>>& values,
                const std::vector<long double>* analytic = nullptr) {
  c.warmups = 0; c.repetitions = 1;
  CpuReference reference(c);
  reference.add_chunk(0, values.data(), values.size());
  auto expected = reference.finish();
  if (analytic && expected != *analytic) throw std::runtime_error("analytic fixture mismatch");
  std::vector<cuda::std::complex<double>> staging;
  for (auto z : values) staging.emplace_back(z.real(), z.imag());
  DeviceState state(c.state_size());
  Stream stream;
  std::vector<double> global_result;
  for (auto storage : {AccumulatorStorage::global, AccumulatorStorage::shared})
  for (auto layout : {GlobalLayout::thread_major, GlobalLayout::bin_major}) {
    if (storage == AccumulatorStorage::shared) {
      try { reduction_resources(c, storage); }
      catch (const Unsupported& e) {
        const std::string reason = e.what();
        if (reason.find("requested-dynamic=") == std::string::npos)
          throw std::runtime_error("resource rejection lacks requested bytes: " + reason);
        try {
          GlobalReduction rejected(c, layout, storage);
          throw std::runtime_error("shared constructor accepted rejected resource configuration");
        } catch (const Unsupported& rejected) {
          if (std::string(rejected.what()).find("requested-dynamic=") == std::string::npos)
            throw std::runtime_error("constructor rejection lacks resource details");
        }
        std::cout << "shared rejected S=" << c.block_size << " k=" << c.positions.size()
                  << ": " << reason << '\n';
        continue;
      }
    }
    cuda_check(cudaMemcpy(state.data(), staging.data(), allocation_bytes(state.size(), sizeof(staging[0])),
                          cudaMemcpyHostToDevice));
    GlobalReduction reduction(c, layout, storage);
    reduction.poison(stream.value);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      if (repeat == 2) cuda_check(cudaMemsetAsync(state.data(), 0,
        allocation_bytes(state.size(), sizeof(staging[0])), stream.value));
      std::vector<double> actual;
      auto result = reduction.run(state, actual, stream.value);
      if (result.status != Status::success || !result.elapsed_ms)
        throw std::runtime_error(std::string(status_name(result.status)) + ": " + result.reason);
      compare(actual, repeat == 2 ? std::vector<long double>(c.bins(), 0) : expected);
      if (repeat == 0) {
        if (storage == AccumulatorStorage::global) global_result = actual;
        else compare(actual, std::vector<long double>(global_result.begin(), global_result.end()));
      }
    }
  }
}
}
int main(int argc, char**) {
  try {
    const bool quick = argc > 1;
    unsigned cases = 0;
    for (unsigned block : {128u, 256u, 512u}) {
      if (quick && block != 128) continue;
      for (auto fixture : deterministic_fixtures()) {
        fixture.config.block_size = block;
        fixture.config.sm_multiplier = 1;
        check_case(fixture.config, fixture.amplitudes, &fixture.expected); ++cases;
      }
      int device; cuda_check(cudaGetDevice(&device));
      cudaDeviceProp props{}; cuda_check(cudaGetDeviceProperties(&props, device));
      for (Size multiplier : {Size{1}, Size{3}}) {
        Config base; base.block_size = block; base.sm_multiplier = multiplier;
        const Size threads = checked_mul(launch_blocks(base, props.multiProcessorCount), block);
        unsigned tail_q = 8;
        while (power_of_two(tail_q) <= threads) ++tail_q;
        for (unsigned q : {8u, tail_q}) {
          if (multiplier == 3 && q == tail_q && power_of_two(q) % threads == 0)
            throw std::runtime_error("tail case must have a grid-stride remainder");
          std::vector<std::complex<double>> values(power_of_two(q));
          for (Size j = 0; j < values.size(); ++j)
            values[j] = {double(int(j % 19) - 9) / 7.0, double(int(j % 23) - 11) / 13.0};
          for (unsigned k = 1; k <= 8; ++k) {
            if (quick && k != 1 && k != 3 && k != 6 && k != 8) continue;
            for (bool reverse : {false, true}) {
              Config c = base; c.qubits = q; c.positions.clear();
              // A runtime permutation with gaps; reverse to check ordered bit mapping.
              for (unsigned t = 0; t < k; ++t)
                c.positions.push_back(t % 2 == 0 ? q - 1 - t/2 : t/2);
              if (reverse) std::reverse(c.positions.begin(), c.positions.end());
              check_case(c, values); ++cases;
            }
          }
        }
      }
    }
    // Resource/capacity rejection must occur before allocation or execution.
    for (unsigned k : {1u, 3u, 4u, 8u}) {
      Config probe; probe.qubits = 8; probe.positions.clear();
      for (unsigned bit = 0; bit < k; ++bit) probe.positions.push_back(bit);
      try {
        GlobalReduction rejected(probe, GlobalLayout::thread_major, AccumulatorStorage::registers);
        throw std::runtime_error("unverified register kernel was accepted");
      } catch (const Unsupported& e) {
        const std::string reason = e.what();
        if (reason.find(k <= 3 ? "residency unverified" : "capacity=8") == std::string::npos)
          throw std::runtime_error("unexpected register rejection: " + reason);
        std::cout << "register k=" << k << ": " << reason << '\n';
      }
    }
    // Exercise nonzero warmups and measured repetitions, using generated input.
    Config c; c.qubits = 8; c.positions = {7, 2, 0}; c.warmups = 2; c.repetitions = 3;
    auto state = generate_state(c);
    auto reference = reference_device_state(c, state);
    for (auto storage : {AccumulatorStorage::global, AccumulatorStorage::shared})
    for (auto layout : {GlobalLayout::thread_major, GlobalLayout::bin_major}) {
      GlobalReduction reduction(c, layout, storage);
      std::vector<double> output;
      auto result = reduction.run(state, output);
      if (result.status != Status::success) throw std::runtime_error(result.reason);
      compare(output, reference);
    }
    std::cout << cases << " cases passed for global and supported shared layouts, dirty workspace reuse and changed input; timing passed\n";
  } catch (const Unsupported& e) {
    std::cerr << "unsupported: " << e.what() << '\n'; return 77;
  } catch (const std::exception& e) {
    std::cerr << "failure: " << e.what() << '\n'; return 1;
  }
}
