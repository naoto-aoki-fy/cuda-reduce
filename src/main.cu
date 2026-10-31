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
    for (auto layout : {measurement::GlobalLayout::thread_major, measurement::GlobalLayout::bin_major}) {
      measurement::Result result;
      try {
        measurement::GlobalReduction reduction(config, layout);
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
        }
      } catch (const measurement::Unsupported& e) {
        result.status = measurement::Status::unsupported;
        result.reason = e.what();
      }
      std::cout << measurement::layout_name(layout) << ": " << measurement::status_name(result.status);
      if (!result.reason.empty()) std::cout << " (" << result.reason << ')';
      if (result.elapsed_ms) std::cout << "; mean ms: " << *result.elapsed_ms;
      std::cout << '\n';
      failed |= result.status == measurement::Status::failure;
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
