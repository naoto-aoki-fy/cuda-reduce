#include "measurement/core.hpp"
#include "measurement/gather.hpp"
#include <algorithm>
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
    for (Size grid : {32,64,128,256,512,1024}) {
      Config c; c.grid_blocks = grid;
      require(launch_blocks(c, 7) == grid && launch_blocks(c, 108) == grid);
      c.sm_multiplier = std::numeric_limits<Size>::max();
      require(launch_blocks(c, 108) == grid); // No unused multiplier overflow.
    }
    { Config c; c.grid_blocks = 0;
      rejects<std::invalid_argument>([&] { c.validate(); });
      c.grid_blocks = Size{1} << 31;
      rejects<std::overflow_error>([&] { launch_blocks(c, 7); });
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
    // Full-preset measurement counts, without allocating the q=31 state.
    for (unsigned k = 9; k <= 16; ++k) {
      for (const auto& band : {"low", "middle", "high"})
      for (bool reverse : {false, true}) {
        Config c; c.qubits = 31; c.positions = contiguous_positions(31, k, band);
        require(c.positions.size() == k);
        const unsigned start = std::string(band) == "low" ? 0 :
          std::string(band) == "middle" ? (31-k)/2 : 31-k;
        for (unsigned t = 0; t < k; ++t) require(c.positions[t] == start+t);
        if (reverse) std::reverse(c.positions.begin(), c.positions.end());
        c.validate();
        require(c.state_size() == 2147483648ULL && c.bins() == power_of_two(k));
        GatherMap map(c);
        OffsetIterator offsets{{static_cast<LogicalOffset>(map.length)}, 0};
        require(offsets[static_cast<LogicalOffset>(c.bins())] == 2147483648LL);
        for (Size bin = 0; bin < c.bins(); ++bin) {
          Size measured = 0;
          for (unsigned t = 0; t < k; ++t)
            measured |= ((bin >> t) & Size{1}) << c.positions[t];
          for (Size offset : {Size{0}, map.length - 1}) {
            const Size j = map(static_cast<LogicalOffset>(bin * map.length + offset));
            const Size remaining_mask = (c.state_size() - 1) & ~map.measured_mask;
            require(j == (measured | (offset ? remaining_mask : 0)));
            require(bin_for(j, c.positions) == bin);
          }
        }
      }
      // Bounded full-input CPU reference, including all 65536 bins at k=16.
      Config c; c.qubits = k; c.positions = contiguous_positions(k, k, "low");
      std::reverse(c.positions.begin(), c.positions.end());
      std::vector<std::complex<double>> values(c.state_size(), {1, 0});
      CpuReference reference(c);
      reference.add_chunk(0, values.data(), values.size());
      require(reference.finish() == std::vector<long double>(c.bins(), 1));
    }
    {
      Config c; c.qubits = 31; c.positions.clear();
      for (unsigned t = 0; t < 17; ++t) c.positions.push_back(t+1);
      rejects<std::invalid_argument>([&] { c.validate(); });
      rejects<std::invalid_argument>([&] { bin_for(0, c.positions); });
      rejects<std::invalid_argument>([] { contiguous_positions(31, 17, "low"); });
      rejects<std::invalid_argument>([] { contiguous_positions(15, 16, "high"); });
    }
    // Independent gather bijection and per-bin CPU comparison for every small k.
    for (unsigned q = 1; q <= 8; ++q)
    for (unsigned k = 1; k <= q; ++k) {
      std::vector<std::vector<unsigned>> sets;
      for (const auto& band : {"low", "middle", "high"})
        sets.push_back(contiguous_positions(q, k, band));
      std::vector<unsigned> scattered;
      for (unsigned t = 0; t < k; ++t)
        scattered.push_back(t % 2 ? t/2 : q - 1 - t/2);
      sets.push_back(scattered);
      for (auto positions : sets) for (bool reverse : {false, true}) {
        if (reverse) std::reverse(positions.begin(), positions.end());
        Config c; c.qubits = q; c.positions = positions;
        GatherMap map(c);
        std::vector<bool> seen(c.state_size(), false);
        std::vector<std::complex<double>> values(c.state_size());
        for (Size j = 0; j < values.size(); ++j) values[j] = {double(j % 7), double(j % 11)};
        CpuReference reference(c);
        reference.add_chunk(0, values.data(), values.size());
        const auto expected = reference.finish();
        std::vector<long double> actual(c.bins(), 0);
        for (Size v = 0; v < c.state_size(); ++v) {
          const Size j = map(static_cast<LogicalOffset>(v));
          require(j < c.state_size() && !seen[j]);
          seen[j] = true;
          require(bin_for(j, positions) == v / map.length);
          actual[v / map.length] += std::norm(values[j]);
        }
        require(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }));
        require(actual == expected);
      }
    }
    // Exercise signed-32-bit boundary arithmetic without allocating a state.
    for (unsigned q : {31u, 32u}) {
      Config c; c.qubits = q; c.positions = {30, 0, 15};
      GatherMap map(c);
      OffsetIterator offsets{{static_cast<LogicalOffset>(map.length)}, 0};
      require(offsets[static_cast<LogicalOffset>(c.bins())] == static_cast<LogicalOffset>(c.state_size()));
      for (LogicalOffset v : {(LogicalOffset{1} << 31) - 1, LogicalOffset{1} << 31,
                              (LogicalOffset{1} << 31) + 1}) {
        LogicalIterator<GatherMap, Size> it{map, v};
        require((it + 7) - it == 7 && it[-1] == map(v - 1));
        require((it + 1) > it && (it - 1) < it);
        if (Size(v) >= c.state_size()) continue;
        Size j = *it, remaining = 0;
        unsigned t = 0;
        for (unsigned bit = 0; bit < q; ++bit)
          if (!(map.measured_mask & (Size{1} << bit)))
            remaining |= ((j >> bit) & Size{1}) << t++;
        require(bin_for(j, c.positions) * map.length + remaining == Size(v));
      }
    }
    std::cout << "host checks passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
