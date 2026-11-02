#pragma once
#include "measurement/core.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace measurement {
struct Options {
  Config config;
  std::string preset = "single", prefix = "benchmark";
  std::vector<unsigned> blocks;
  std::optional<Size> profile_case;
  bool validate = true;
  bool device_info = false, stream_rows = false;
  std::string case_plan;
  Size workspace_limit = ~Size{0}, memory_reserve = 0, run_budget_ms = 0;
};
inline Options options(int argc, char** argv) {
  Options o;
  std::vector<const char*> args{argv[0]};
  bool qubits = false, positions = false, multiplier = false;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    std::string key = argv[i], value = argv[i+1];
    if (key == "--case-plan") o.case_plan = value;
    else if (key == "--stream-rows") o.stream_rows = value == "1";
    else if (key == "--device-info") o.device_info = value == "1";
    else if (key == "--workspace-limit-bytes" || key == "--memory-reserve-bytes" || key == "--run-budget-ms") {
      const char* a[] = {argv[0], "--seed", argv[i+1]};
      auto n = parse_config(3, a).seed;
      if (key == "--workspace-limit-bytes") o.workspace_limit = n;
      else if (key == "--run-budget-ms") o.run_budget_ms = n; else o.memory_reserve = n;
    }
    else if (key == "--preset") o.preset = value;
    else if (key == "--results") o.prefix = value;
    else if (key == "--validation") {
      if (value != "full" && value != "skip") throw std::invalid_argument("validation: full or skip");
      o.validate = value == "full";
    } else if (key == "--profile-case") {
      const char* a[] = {argv[0], "--seed", argv[i+1]};
      o.profile_case = parse_config(3, a).seed;
    } else if (key == "--block-sizes") {
      std::string token;
      std::istringstream list(value);
      while (std::getline(list, token, ',')) {
        const char* a[] = {argv[0], "--block-size", token.c_str()};
        unsigned block = parse_config(3, a).block_size;
        if (std::find(o.blocks.begin(), o.blocks.end(), block) != o.blocks.end())
          throw std::invalid_argument("duplicate block size");
        o.blocks.push_back(block);
      }
      if (o.blocks.empty() || value.back() == ',') throw std::invalid_argument("empty block size");
    } else {
      qubits |= key == "--qubits"; positions |= key == "--positions";
      multiplier |= key == "--sm-multiplier";
      args.push_back(argv[i]); args.push_back(argv[i+1]);
    }
  }
  o.config = parse_config(static_cast<int>(args.size()), args.data());
  if (o.preset != "full" && o.preset != "full-k1-6" && o.preset != "smoke" && o.preset != "single")
    throw std::invalid_argument("preset: single, smoke, full or full-k1-6");
  if (o.preset != "single" && (positions || multiplier))
    throw std::invalid_argument("matrix preset owns positions and SM multipliers");
  if (o.preset == "full" || o.preset == "full-k1-6") {
    if (qubits) throw std::invalid_argument("full preset fixes q=31; use smoke for smaller input");
    o.config.qubits = 31;
  }
  if (o.blocks.empty()) o.blocks.push_back(o.config.block_size);
  o.config.validate();
  return o;
}
struct Case {
  Config c;
  std::size_t reference;
  std::string band;
  BlockPrimitive primitive;
  AccumulatorStorage storage;
  GlobalLayout layout;
  bool segmented;
};
struct Matrix {
  std::vector<Config> references;
  std::vector<Case> cases;
};
inline Matrix matrix(const Options& o) {
  Matrix m;
  auto add = [&](Config c, const std::string& band) {
    const auto ref = m.references.size();
    m.references.push_back(c);
    for (unsigned block : o.blocks)
    for (Size mult : o.preset == "single" ? std::vector<Size>{c.sm_multiplier} : std::vector<Size>{1,2,4,8,16})
    for (auto primitive : {BlockPrimitive::custom, BlockPrimitive::cub})
    for (auto storage : {AccumulatorStorage::registers, AccumulatorStorage::shared, AccumulatorStorage::global})
    for (auto layout : {GlobalLayout::thread_major, GlobalLayout::bin_major}) {
      if (storage == AccumulatorStorage::registers && layout == GlobalLayout::bin_major) continue;
      c.block_size = block; c.sm_multiplier = mult;
      m.cases.push_back({c, ref, band, primitive, storage, layout, false});
    }
    // No repetition across irrelevant custom tuning dimensions.
    m.cases.push_back({m.references[ref], ref, band, BlockPrimitive::cub,
      AccumulatorStorage::global, GlobalLayout::thread_major, true});
  };
  if (!o.case_plan.empty()) {
    // Internal, validated planner interchange: one q/seed per process.
    std::ifstream input(o.case_plan);
    if (!input) throw std::invalid_argument("cannot open case plan");
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream fields(line);
      std::string positions, band, primitive, storage, layout, strategy, extra;
      Config c = o.config;
      if (!(fields >> positions >> band >> primitive >> storage >> layout >> strategy
                   >> c.block_size >> c.sm_multiplier) || fields >> extra)
        throw std::invalid_argument("malformed case plan row");
      const char* a[] = {"plan", "--positions", positions.c_str()};
      // parse positions with the actual q, rather than the parser's default q.
      std::string q = std::to_string(c.qubits);
      const char* args[] = {a[0], "--qubits", q.c_str(), a[1], a[2]};
      c.positions = parse_config(5, args).positions;
      c.validate();
      if (primitive != "custom" && primitive != "cub" && primitive != "segmented")
        throw std::invalid_argument("invalid plan method");
      if (storage != "global" && storage != "shared" && storage != "registers")
        throw std::invalid_argument("invalid plan storage");
      if (layout != "thread_major" && layout != "bin_major") throw std::invalid_argument("invalid plan layout");
      if (strategy != "two_kernel") throw std::invalid_argument("invalid plan strategy");
      std::size_t ref = 0;
      while (ref < m.references.size() && m.references[ref].positions != c.positions) ++ref;
      if (ref == m.references.size()) m.references.push_back(c);
      m.cases.push_back({c, ref, band, primitive == "custom" ? BlockPrimitive::custom : BlockPrimitive::cub,
        storage == "global" ? AccumulatorStorage::global : storage == "shared" ? AccumulatorStorage::shared : AccumulatorStorage::registers,
        layout == "thread_major" ? GlobalLayout::thread_major : GlobalLayout::bin_major,
        primitive == "segmented"});
    }
    if (m.cases.empty()) throw std::invalid_argument("empty case plan");
  }
  else if (o.preset == "single") add(o.config, "ordered");
  else for (unsigned k = 1;
            k <= std::min(o.preset == "full-k1-6" ? 6u :
                          o.preset == "full" ? max_measurement_qubits : 8u, o.config.qubits); ++k)
    for (const auto& band : {"low", "middle", "high"}) {
      Config c = o.config;
      c.positions = contiguous_positions(c.qubits, k, band);
      add(c, band);
    }
  return m;
}

// Relative work units, not milliseconds. Keep in sync with scheduling_cost in
// tools/benchmark_matrix.py. A fixed 128-SM proxy avoids device calibration.
inline long double scheduling_cost(const Case& x) {
  const long double n = x.c.state_size(), bins = x.c.bins();
  const long double k = x.c.positions.size();
  long double work;
  if (x.segmented) {
    // Virtual gather computes k-bit addresses for each input; strided reads.
    work = n * (24 + 4 * k) + 16 * bins;
  } else {
    const long double threads = 128.L * x.c.sm_multiplier * x.c.block_size;
    const long double storage = x.storage == AccumulatorStorage::global ? 16 :
                                x.storage == AccumulatorStorage::shared ? 4 : 1;
    const long double primitive = x.primitive == BlockPrimitive::cub ? 1.1L : 1;
    work = n * (16 + 2 * k + storage) +
           threads * bins * (storage + 8 * primitive) +
           16.L * 128 * x.c.sm_multiplier * bins;
  }
  return work * (static_cast<long double>(x.c.warmups) + x.c.repetitions);
}
inline std::vector<std::size_t> execution_order(const Matrix& m, const Options& o) {
  std::vector<std::size_t> order(m.cases.size());
  std::iota(order.begin(), order.end(), 0);
  // Planner rows are already scheduled by Python; their indices are the child
  // protocol IDs. Never change the matrix itself or profiling case selection.
  if (o.case_plan.empty())
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return scheduling_cost(m.cases[a]) < scheduling_cost(m.cases[b]);
    });
  return order;
}

} // namespace measurement
