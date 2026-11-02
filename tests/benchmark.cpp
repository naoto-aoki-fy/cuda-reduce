#include "measurement/benchmark.hpp"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <tuple>
using namespace measurement;
int main() {
  const char* full[] = {"test", "--preset", "full"};
  auto o = options(3, const_cast<char**>(full));
  auto m = matrix(o);
  assert(o.config.qubits == 31 && m.references.size() == 48 && m.cases.size() == 4848);
  std::size_t segmented = 0;
  for (const auto& x : m.cases) segmented += x.segmented;
  assert(segmented == 48);
  assert(m.references.front().positions == std::vector<unsigned>{0});
  assert(m.references.back().positions == contiguous_positions(31,16,"high"));
  // The restricted preset must be exactly the k <= 6 prefix of full, including
  // every tuning dimension and all timing/input settings, with or without overrides.
  for (const char* block_sizes : {"256", "32,64,128,256,512,1024"}) {
    const char* original[] = {"test", "--preset", "full", "--block-sizes", block_sizes,
      "--repetitions", "2", "--warmups", "4", "--seed", "42"};
    const char* restricted[] = {"test", "--preset", "full-k1-6", "--block-sizes", block_sizes,
      "--repetitions", "2", "--warmups", "4", "--seed", "42"};
    auto full_matrix = matrix(options(11, const_cast<char**>(original)));
    auto limited = matrix(options(11, const_cast<char**>(restricted)));
    auto fields = [](const Config& c) {
      return std::tie(c.qubits, c.positions, c.block_size, c.sm_multiplier,
                      c.repetitions, c.warmups, c.seed, c.chunk_elements, c.case_budget_ms);
    };
    assert(limited.references.size() == 18);
    assert(limited.cases.size() == (std::string(block_sizes) == "256" ? 1818 : 10818));
    std::size_t expected = 0;
    for (const auto& x : full_matrix.cases) {
      if (x.c.positions.size() > 6) continue;
      const auto& y = limited.cases.at(expected++);
      assert(fields(x.c) == fields(y.c));
      assert(std::tie(x.reference, x.band, x.primitive, x.storage, x.layout, x.method, x.segmented)
          == std::tie(y.reference, y.band, y.primitive, y.storage, y.layout, y.method, y.segmented));
    }
    assert(expected == limited.cases.size());
    for (std::size_t i = 0; i < limited.references.size(); ++i) {
      assert(fields(limited.references[i]) == fields(full_matrix.references[i]));
      assert(limited.references[i].positions.size() == i / 3 + 1);
    }
  }
  const char* blocks[] = {"test", "--preset", "full", "--block-sizes", "128,256,512"};
  assert(matrix(options(5, const_cast<char**>(blocks))).cases.size() == 14448);
  const char* all_blocks[] = {"test", "--preset", "full", "--block-sizes", "32,64,128,256,512,1024"};
  auto all = matrix(options(5, const_cast<char**>(all_blocks)));
  assert(all.cases.size() == 28848 && all.references.size() == 48);
  std::size_t baseline_count = 0;
  for (const auto& x : all.cases) baseline_count += x.segmented;
  assert(baseline_count == 48);
  // Every tuning combination is identical across block sizes.
  for (unsigned block : {32u, 64u, 128u, 256u, 512u, 1024u}) {
    std::size_t count = 0;
    for (const auto& x : all.cases) if (!x.segmented && x.c.block_size == block) {
      assert(x.c.qubits == 31 && x.c.seed == o.config.seed);
      assert(x.c.repetitions == o.config.repetitions && x.c.warmups == o.config.warmups);
      ++count;
    }
    assert(count == 4800);
  }
  const char* smoke[] = {"test", "--preset", "smoke", "--qubits", "3"};
  assert(matrix(options(5, const_cast<char**>(smoke))).references.size() == 9);
  const char* single[] = {"test", "--positions", "7,1"};
  m = matrix(options(3, const_cast<char**>(single)));
  assert(m.references.size() == 1 && m.cases.size() == 21);
  assert(m.references[0].positions == (std::vector<unsigned>{7,1}));
  for (const auto* key : {"--qubits", "--positions", "--sm-multiplier"}) {
    for (const auto* preset : {"full", "full-k1-6"}) {
      const char* bad[] = {"test", "--preset", preset, key, "1"};
      bool rejected = false;
      try { options(5, const_cast<char**>(bad)); } catch (const std::invalid_argument&) { rejected=true; }
      assert(rejected);
    }
  }
  // Planner interchange retains ordered positions and deduplicates references.
  const std::string path = "build/host-test-plan.tsv";
  { std::ofstream f(path); f << "7,1 ordered custom global thread_major two_kernel 128 2\n"
                              "1,7 reversed cub shared bin_major two_kernel 256 4\n"
                              "7,1 ordered segmented global thread_major two_kernel 256 1\n"; }
  const char* planned[] = {"test", "--case-plan", path.c_str()};
  m = matrix(options(3, const_cast<char**>(planned)));
  assert(m.references.size() == 2 && m.cases.size() == 3);
  assert(m.cases[0].c.positions != m.cases[1].c.positions);
  assert(m.cases[2].segmented && m.cases[0].reference == m.cases[2].reference);
  std::remove(path.c_str());
  std::cout << "benchmark host matrix checks passed\n";
}
