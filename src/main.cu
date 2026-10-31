#include "measurement/cuda.hpp"
#include <cmath>
#include <iostream>
int main(int argc, char** argv) {
  try {
    auto config = measurement::parse_config(argc, argv);
    auto state = measurement::generate_state(config);
    auto bins = measurement::reference_device_state(config, state);
    std::cout << "CPU reference: success\n";
    for (std::size_t b = 0; b < bins.size(); ++b) std::cout << b << ' ' << bins[b] << '\n';
    bool failed = false;
    for (auto storage : {measurement::AccumulatorStorage::global, measurement::AccumulatorStorage::shared,
                         measurement::AccumulatorStorage::registers})
    for (auto layout : {measurement::GlobalLayout::thread_major, measurement::GlobalLayout::bin_major}) {
      if (storage == measurement::AccumulatorStorage::registers && layout == measurement::GlobalLayout::bin_major) continue;
      std::vector<double> two_kernel_output;
      for (auto method : {measurement::ReductionMethod::two_kernel, measurement::ReductionMethod::cooperative}) {
        const std::string label = std::string(method == measurement::ReductionMethod::cooperative ?
          "cooperative " : "two-kernel ") + std::string(measurement::storage_name(storage)) + " " +
          (storage == measurement::AccumulatorStorage::registers ? "" :
           layout == measurement::GlobalLayout::thread_major ? "thread-major" : "bin-major");
        measurement::Result result;
        try {
          std::cout << label << " resources: " << measurement::resource_description(
            measurement::reduction_resources(config, storage, method)) << '\n';
          measurement::GlobalReduction reduction(config, layout, storage, method);
          std::vector<double> output;
          result = reduction.run(state, output);
          if (result.status == measurement::Status::success) {
            for (std::size_t b = 0; b < bins.size(); ++b) {
              if (!std::isfinite(output[b]) ||
                  std::abs(static_cast<long double>(output[b]) - bins[b]) > 1e-12L + 1e-11L * std::abs(bins[b])) {
                result.status = measurement::Status::failure;
                result.reason = "CPU reference mismatch at bin " + std::to_string(b);
                result.elapsed_ms.reset();
                break;
              }
            }
            if (result.status == measurement::Status::success) {
              if (method == measurement::ReductionMethod::two_kernel) two_kernel_output = output;
              else if (two_kernel_output.size() != output.size()) {
                result.status = measurement::Status::failure;
                result.reason = "matching two-kernel result unavailable";
                result.elapsed_ms.reset();
              } else for (std::size_t b = 0; b < output.size(); ++b) {
                if (!std::isfinite(two_kernel_output[b]) ||
                    std::abs(output[b] - two_kernel_output[b]) > 1e-12 + 1e-11 * std::abs(two_kernel_output[b])) {
                  result.status = measurement::Status::failure;
                  result.reason = "matching two-kernel mismatch at bin " + std::to_string(b);
                  result.elapsed_ms.reset();
                  break;
                }
              }
            }
          }
        } catch (const measurement::Unsupported& e) {
          result.status = measurement::Status::unsupported;
          result.reason = e.what();
        }
        std::cout << label << ": " << measurement::status_name(result.status);
        if (!result.reason.empty()) std::cout << " (" << result.reason << ')';
        if (result.elapsed_ms) std::cout << "; mean ms: " << *result.elapsed_ms;
        std::cout << '\n';
        failed |= result.status == measurement::Status::failure;
      }
    }
    std::cout << "profiling metrics: unavailable\n";
    return failed ? 1 : 0;
  } catch (const measurement::Unsupported& e) {
    std::cout << "unsupported: " << e.what() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "failure: " << e.what() << '\n';
    return 1;
  }
}
