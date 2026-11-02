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
  assert(o.config.qubits == 31 && m.references.size() == 48 && m.cases.size() == 2448);
  std::size_t segmented = 0;
  for (const auto& x : m.cases) segmented += x.segmented;
  assert(segmented == 48);
  // Scheduling never renumbers the matrix used by --profile-case. Every case
  // appears exactly once, with ascending estimated work and stable ties.
  for (const char* preset : {"full", "full-k1-6", "full-k4", "smoke", "single"}) {
    const char* args[] = {"test", "--preset", preset};
    auto opts = options(3, const_cast<char**>(args));
    auto planned = matrix(opts);
    auto order = execution_order(planned, opts);
    assert(order == execution_order(planned, opts));
    auto ids = order;
    std::sort(ids.begin(), ids.end());
    for (std::size_t i = 0; i < ids.size(); ++i) assert(ids[i] == i);
    bool changed = false;
    for (std::size_t i = 1; i < order.size(); ++i) {
      const auto a = scheduling_cost(planned.cases[order[i-1]]);
      const auto b = scheduling_cost(planned.cases[order[i]]);
      assert(a <= b);
      if (a == b) assert(order[i-1] < order[i]);
      changed |= order[i] != i;
    }
    assert(changed);
    auto cheap = planned.cases.front(), costly = cheap;
    costly.c.qubits += 1;
    assert(scheduling_cost(cheap) < scheduling_cost(costly));
    costly = cheap; costly.c.sm_multiplier *= 2;
    assert(scheduling_cost(cheap) < scheduling_cost(costly));
    costly = cheap; costly.c.block_size *= 2;
    assert(scheduling_cost(cheap) < scheduling_cost(costly));
    costly = cheap; costly.c.positions.push_back(1);
    assert(scheduling_cost(cheap) < scheduling_cost(costly));
  }
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
    assert(limited.cases.size() == (std::string(block_sizes) == "256" ? 918 : 5418));
    std::size_t expected = 0;
    for (const auto& x : full_matrix.cases) {
      if (x.c.positions.size() > 6) continue;
      const auto& y = limited.cases.at(expected++);
      assert(fields(x.c) == fields(y.c));
      assert(std::tie(x.reference, x.band, x.primitive, x.storage, x.layout, x.segmented)
          == std::tie(y.reference, y.band, y.primitive, y.storage, y.layout, y.segmented));
    }
    assert(expected == limited.cases.size());
    for (std::size_t i = 0; i < limited.references.size(); ++i) {
      assert(fields(limited.references[i]) == fields(full_matrix.references[i]));
      assert(limited.references[i].positions.size() == i / 3 + 1);
    }
  }
  // full-k4 is the exact k=4 subset, with locally renumbered references.
  // Check defaults and all block sizes with timing/resource/validation overrides.
  for (bool overrides : {false, true}) {
    std::vector<const char*> args{"test", "--preset", "full"};
    if (overrides) {
      const std::vector<const char*> extra{"--block-sizes", "32,64,128,256,512,1024",
        "--repetitions", "2", "--warmups", "4", "--seed", "42", "--chunk-elements", "128",
        "--case-budget-ms", "100", "--run-budget-ms", "200", "--validation", "skip",
        "--workspace-limit-bytes", "1024", "--memory-reserve-bytes", "512"};
      args.insert(args.end(), extra.begin(), extra.end());
    }
    auto original = options(static_cast<int>(args.size()), const_cast<char**>(args.data()));
    args[2] = "full-k4";
    auto restricted = options(static_cast<int>(args.size()), const_cast<char**>(args.data()));
    assert(restricted.config.qubits == 31);
    assert(std::tie(original.blocks, original.validate, original.workspace_limit,
                    original.memory_reserve, original.run_budget_ms, original.prefix)
        == std::tie(restricted.blocks, restricted.validate, restricted.workspace_limit,
                    restricted.memory_reserve, restricted.run_budget_ms, restricted.prefix));
    auto full_matrix = matrix(original), limited = matrix(restricted);
    auto fields = [](const Config& c) {
      return std::tie(c.qubits, c.positions, c.block_size, c.sm_multiplier,
                      c.repetitions, c.warmups, c.seed, c.chunk_elements, c.case_budget_ms);
    };
    assert(limited.references.size() == 3);
    assert(limited.cases.size() == (overrides ? 903 : 153));
    std::size_t expected = 0, register_requests = 0, segmented_requests = 0;
    for (const auto& x : full_matrix.cases) {
      if (x.c.positions.size() != 4) continue;
      const auto& y = limited.cases.at(expected++);
      assert(fields(x.c) == fields(y.c));
      assert(std::tie(x.band, x.primitive, x.storage, x.layout, x.segmented)
          == std::tie(y.band, y.primitive, y.storage, y.layout, y.segmented));
      assert(y.reference == x.reference - 9);
      assert(fields(full_matrix.references.at(x.reference)) == fields(limited.references.at(y.reference)));
      register_requests += y.storage == AccumulatorStorage::registers;
      segmented_requests += y.segmented;
    }
    assert(expected == limited.cases.size());
    // k=4 register requests exceed the eight-bin capacity; retain all of them.
    assert(register_requests == (overrides ? 180 : 30));
    assert(segmented_requests == 3);
    std::size_t band_index = 0;
    for (const char* band : {"low", "middle", "high"})
      assert(limited.references[band_index++].positions == contiguous_positions(31, 4, band));
  }
  const char* blocks[] = {"test", "--preset", "full", "--block-sizes", "128,256,512"};
  assert(matrix(options(5, const_cast<char**>(blocks))).cases.size() == 7248);
  const char* all_blocks[] = {"test", "--preset", "full", "--block-sizes", "32,64,128,256,512,1024"};
  auto all = matrix(options(5, const_cast<char**>(all_blocks)));
  assert(all.cases.size() == 14448 && all.references.size() == 48);
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
    assert(count == 2400);
  }
  const char* smoke[] = {"test", "--preset", "smoke", "--qubits", "3"};
  assert(matrix(options(5, const_cast<char**>(smoke))).references.size() == 9);
  const char* single[] = {"test", "--positions", "7,1"};
  m = matrix(options(3, const_cast<char**>(single)));
  assert(m.references.size() == 1 && m.cases.size() == 11);
  assert(m.references[0].positions == (std::vector<unsigned>{7,1}));
  for (const auto* key : {"--qubits", "--positions", "--sm-multiplier"}) {
    for (const auto* preset : {"full", "full-k1-6", "full-k4"}) {
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
  assert(execution_order(m, options(3, const_cast<char**>(planned))) ==
         (std::vector<std::size_t>{0,1,2}));
  assert(m.cases[0].c.positions != m.cases[1].c.positions);
  assert(m.cases[2].segmented && m.cases[0].reference == m.cases[2].reference);
  // Obsolete strategy values must fail parsing rather than generate launch cases.
  for (const char* primitive : {"custom", "cub", "segmented"}) {
    { std::ofstream f(path); f << "7,1 ordered " << primitive
        << " global thread_major cooperative 128 2\n"; }
    bool rejected = false;
    try { matrix(options(3, const_cast<char**>(planned))); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  std::remove(path.c_str());
  std::cout << "benchmark host matrix checks passed\n";
}
