#include "measurement/core.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace measurement;
void require(bool value) { if (!value) throw std::runtime_error("host check failed"); }
template<class Exception, class F> void rejects(F f) {
  try { f(); } catch (const Exception&) { return; }
  throw std::runtime_error("expected rejection");
}
int main() {
  try {
    require(power_of_two(31) == 2147483648ULL);
    require(allocation_bytes(power_of_two(31), 16) == 34359738368ULL);
    require(workspace_elements(7, 512, 256) == 917504);
    rejects<std::overflow_error>([] { checked_mul(std::numeric_limits<Size>::max(), 2); });
    rejects<std::overflow_error>([] { workspace_elements(Size{1} << 60, 512, 256); });
    rejects<std::overflow_error>([] { allocation_bytes(Size{1} << 61, 8); });
    rejects<std::overflow_error>([] { checked_add(std::numeric_limits<Size>::max(), 1); });
    for (unsigned block : {128u, 256u, 512u}) {
      Config c; c.block_size = block;
      c.validate(); require(launch_blocks(c, 7) == 56);
    }
    for (unsigned block : {0u, 32u, 129u, 1024u}) {
      Config c; c.block_size = block;
      rejects<std::invalid_argument>([&] { c.validate(); });
    }
    for (auto fixture : deterministic_fixtures()) {
      CpuReference reference(fixture.config);
      reference.add_chunk(0, fixture.amplitudes.data(), 3);
      reference.add_chunk(3, fixture.amplitudes.data() + 3, 5);
      require(reference.finish() == fixture.expected);
    }
    for (unsigned k = 1; k <= 8; ++k) {
      std::vector<unsigned> positions;
      for (unsigned t = 0; t < k; ++t) positions.push_back((3*t + 1) % 8);
      for (Size j = 0; j < 256; ++j) {
        Size expected = 0;
        for (unsigned t = 0; t < k; ++t)
          if (j & (Size{1} << positions[t])) expected += Size{1} << t;
        require(bin_for(j, positions) == expected);
      }
    }
    std::cout << "host checks passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
