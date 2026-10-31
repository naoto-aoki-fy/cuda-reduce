#include "measurement/cuda.hpp"
#include <iostream>
int main(int argc, char** argv) {
  try {
    auto config = measurement::parse_config(argc, argv);
    auto state = measurement::generate_state(config);
    auto bins = measurement::reference_device_state(config, state);
    std::cout << "CPU reference: success\n";
    for (std::size_t b = 0; b < bins.size(); ++b) std::cout << b << ' ' << bins[b] << '\n';
    std::cout << "GPU reductions: not_run (Step 1); profiling metrics: unavailable\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "failure: " << e.what() << '\n';
    return 1;
  }
}
