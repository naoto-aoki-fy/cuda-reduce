#include "measurement/cuda.hpp"
#include "measurement/benchmark.hpp"
#include "build_info.hpp"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

using namespace measurement;
namespace {
using Fields = std::map<std::string, std::string>; // Values are JSON tokens.
std::string json_quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else out << c;
  }
  out << '"';
  return out.str();
}
template<class T> std::string number(T v) {
  std::ostringstream s;
  s << std::setprecision(21) << v;
  return s.str();
}
template<class T> std::string array(const std::vector<T>& values) {
  std::string s = "[";
  for (const auto& v : values) { if (s.size() > 1) s += ','; s += number(v); }
  return s + ']';
}
std::string json(const Fields& f) {
  std::string s = "{";
  for (const auto& p : f) {
    if (s.size() > 1) s += ',';
    s += json_quote(p.first) + ':' + p.second;
  }
  return s + '}';
}
std::string csv(const std::string& s) {
  std::string r = "\"";
  for (char c : s) { if (c == '"') r += '"'; r += c; }
  return r + '"';
}
struct ProfileRange {
  ProfileRange() { cuda_check(cudaProfilerStart()); }
  ~ProfileRange() { cudaProfilerStop(); }
};
struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
std::vector<std::vector<long double>> references(const Matrix& m, const DeviceState& state,
                                               const Options& o) {
  std::vector<CpuReference> cpu;
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < m.references.size(); ++i) {
    if (o.profile_case && m.cases[*o.profile_case].reference != i) continue;
    cpu.emplace_back(m.references[i]); indices.push_back(i);
  }
  Size capacity = std::min(o.config.chunk_elements, state.size());
  std::vector<cuda::std::complex<double>> staging(static_cast<std::size_t>(capacity));
  std::vector<std::complex<double>> host(static_cast<std::size_t>(capacity));
  for (Size offset = 0; offset < state.size();) {
    Size count = std::min(capacity, state.size() - offset);
    cuda_check(cudaMemcpy(staging.data(), state.data() + offset,
      allocation_bytes(count, sizeof(staging[0])), cudaMemcpyDeviceToHost));
    for (Size i = 0; i < count; ++i) host[i] = {staging[i].real(), staging[i].imag()};
    for (auto& ref : cpu) ref.add_chunk(offset, host.data(), count);
    offset += count;
  }
  std::vector<std::vector<long double>> result(m.references.size());
  for (std::size_t i = 0; i < cpu.size(); ++i) result[indices[i]] = cpu[i].finish();
  return result;
}
constexpr long double kAbsoluteTolerance = 1e-12L;
constexpr long double kRelativeTolerance = 1e-11L;
bool validate(Fields& row, const std::vector<double>& output, const std::vector<long double>& ref) {
  if (output.size() != ref.size()) throw std::runtime_error("output bin count mismatch");
  long double max_abs = 0, max_relative = 0, actual_total = 0, ref_total = 0;
  bool ok = true;
  for (std::size_t b = 0; b < ref.size(); ++b) {
    const long double err = std::abs(static_cast<long double>(output[b]) - ref[b]);
    ok &= std::isfinite(output[b]) && err <= kAbsoluteTolerance + kRelativeTolerance * std::abs(ref[b]);
    if (!std::isfinite(output[b])) {
      row["validation_status"] = json_quote("failure");
      row["validation_reason"] = json_quote("nonfinite output at bin " + std::to_string(b));
      return false;
    }
    max_abs = std::max(max_abs, err);
    max_relative = std::max(max_relative, err / std::max(std::abs(ref[b]), kAbsoluteTolerance));
    actual_total += output[b]; ref_total += ref[b];
  }
  long double total_error = std::abs(actual_total - ref_total);
  ok &= total_error <= kAbsoluteTolerance + kRelativeTolerance * std::abs(ref_total);
  row["max_absolute_error"] = number(max_abs);
  row["max_relative_error"] = number(max_relative);
  row["reference_total_norm"] = number(ref_total);
  row["total_norm_absolute_error"] = number(total_error);
  row["validation_status"] = json_quote(ok ? "success" : "failure");
  row["validation_reason"] = json_quote(ok ? "every bin and total norm; entire actual device input" : "bin or total norm tolerance exceeded");
  return ok;
}
void kernel(Fields& row, const KernelResources& r, unsigned stage) {
  auto prefix = "stage" + std::to_string(stage) + '_';
  row[prefix + "binary_architecture"] = number(r.binary_version);
  row[prefix + "ptx_architecture"] = number(r.ptx_version);
  row[prefix + "registers_per_thread"] = number(r.registers_per_thread);
  row[prefix + "local_bytes_per_thread"] = number(r.local_bytes_per_thread);
  row[prefix + "static_shared_bytes_per_block"] = number(r.static_shared_bytes);
  row[prefix + "dynamic_shared_bytes_per_block"] = number(r.dynamic_shared_bytes);
  row[prefix + "active_blocks_per_sm"] = number(r.active_blocks_per_sm);
  row[prefix + "theoretical_occupancy"] = number(r.theoretical_occupancy);
}
Fields row_template(const Case& x, Size id, const Options& o) {
  Fields row;
  for (const auto* field : {"mean_ms", "min_ms", "median_ms", "max_ms", "sample_stddev_ms",
    "input_effective_GB_per_s", "max_absolute_error", "max_relative_error", "reference_total_norm",
    "total_norm_absolute_error", "workspace_bytes", "total_live_device_bytes", "blocks",
    "estimated_workspace_bytes", "compiler_spill_loads", "compiler_spill_stores", "achieved_occupancy", "dram_read_bytes", "dram_write_bytes"})
    row[field] = "null";
  for (unsigned stage : {1u, 2u}) for (const auto* field : {"binary_architecture", "ptx_architecture", "registers_per_thread", "local_bytes_per_thread",
    "static_shared_bytes_per_block", "dynamic_shared_bytes_per_block", "active_blocks_per_sm", "theoretical_occupancy"})
    row["stage" + std::to_string(stage) + '_' + field] = "null";
  row["case_id"] = number(id); row["preset"] = json_quote(o.preset);
  row["N"] = number(x.c.state_size()); row["state_bytes"] = number(allocation_bytes(x.c.state_size(), sizeof(cuda::std::complex<double>)));
  row["q"] = number(x.c.qubits); row["bins"] = number(x.c.bins());
  row["completed_repetitions"] = "0"; row["timing_mode"] = json_quote("single_invocation");
  row["k"] = number(x.c.positions.size()); row["positions"] = array(x.c.positions); row["band"] = json_quote(x.band);
  row["seed"] = number(x.c.seed); row["chunk_elements"] = number(x.c.chunk_elements);
  row["repetitions"] = number(x.c.repetitions); row["warmups"] = number(x.c.warmups);
  row["method"] = json_quote(x.segmented ? "CUB DeviceSegmentedReduce" : x.primitive == BlockPrimitive::cub ? "CUB BlockReduce" : "custom");
  row["strategy"] = x.segmented ? "null" : json_quote("two_kernel");
  row["storage"] = x.segmented ? "null" : json_quote(x.storage == AccumulatorStorage::registers ? "registers" : x.storage == AccumulatorStorage::shared ? "shared" : "global");
  row["layout"] = x.segmented || x.storage == AccumulatorStorage::registers ? "null" : json_quote(x.layout == GlobalLayout::thread_major ? "thread_major" : "bin_major");
  row["block_size"] = x.segmented ? "null" : number(x.c.block_size);
  row["sm_multiplier"] = x.segmented ? "null" : number(x.c.sm_multiplier);
  row["status"] = json_quote("not_run"); row["timing_status"] = json_quote("not_run");
  row["validation_status"] = json_quote(o.validate ? "not_run" : "skipped");
  row["validation_reason"] = json_quote(""); row["reason"] = json_quote(""); row["resources"] = "null";
  row["absolute_tolerance"] = number(kAbsoluteTolerance); row["relative_tolerance"] = number(kRelativeTolerance);
  row["raw_elapsed_ms"] = "[]"; row["profiling_run"] = o.profile_case ? "true" : "false";
  row["profiling_status"] = json_quote("not_run");
  row["runtime_metrics_status"] = json_quote("unavailable; separate tools/profile_benchmark.py run required");
  row["compiler_spill_status"] = json_quote("unavailable in runtime API; compiler information in make cuda-evidence build.log");
  return row;
}
class Writer {
  std::ofstream json_, csv_;
  bool first_ = true;
 public:
  explicit Writer(const std::string& prefix, const Fields& metadata) : json_(prefix + ".json"), csv_(prefix + ".csv") {
    if (!json_ || !csv_) throw std::runtime_error("cannot open result files at " + prefix);
    json_ << "{\"metadata\":" << json(metadata) << ",\"results\":[\n";
  }
  void write(const Fields& row) {
    if (first_) {
      bool first = true;
      for (const auto& p : row) { if (!first) csv_ << ','; first = false; csv_ << csv(p.first); }
      csv_ << '\n';
    } else json_ << ",\n";
    first_ = false;
    json_ << json(row); json_.flush();
    bool first = true;
    for (const auto& p : row) { if (!first) csv_ << ','; first = false; csv_ << csv(p.second); }
    csv_ << '\n'; csv_.flush();
    if (!json_ || !csv_) throw std::runtime_error("result write failed");
  }
  void finish(const Fields& summary) {
    json_ << "\n],\"summary\":" << json(summary) << "}\n";
    json_.flush();
    if (!json_) throw std::runtime_error("result finalization failed");
  }
};
}
int main(int argc, char** argv) {
  try {
    const auto run_start = std::chrono::steady_clock::now();
    const auto o = options(argc, argv);
    const auto m = matrix(o);
    if (o.profile_case && *o.profile_case >= m.cases.size()) throw std::invalid_argument("profile case outside matrix");
    Fields metadata{{"build", MEASUREMENT_BUILD_INFO}, {"cub", json_quote(cub_version_description())},
      {"input_generation", json_quote("PHILOX4_32_10; PSEUDO_DEFAULT; offset=0; uniform doubles (0,1]; 2^20 amplitudes/chunk")},
      {"GPU", "null"}, {"compute_capability", "null"}, {"SM_count", "null"},
      {"CUDA_runtime", "null"}, {"CUDA_driver", "null"}, {"cuRAND", "null"}};
    std::string startup_reason;
    Status startup = Status::success;
    std::unique_ptr<Stream> stream;
    std::unique_ptr<DeviceState> state;
    std::vector<std::vector<long double>> refs;
    try {
      int count = 0;
      auto availability = cudaGetDeviceCount(&count);
      if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver ||
          (availability == cudaSuccess && count == 0))
        throw Unsupported("CUDA device/driver unavailable");
      cuda_check(availability);
      int device; cuda_check(cudaGetDevice(&device));
      cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop, device));
      metadata["GPU"] = json_quote(prop.name); metadata["SM_count"] = number(prop.multiProcessorCount);
      metadata["compute_capability"] = json_quote(std::to_string(prop.major) + '.' + std::to_string(prop.minor));
      int runtime, driver, curand;
      cuda_check(cudaRuntimeGetVersion(&runtime)); cuda_check(cudaDriverGetVersion(&driver));
      curand_check(curandGetVersion(&curand));
      metadata["CUDA_runtime"] = number(runtime); metadata["CUDA_driver"] = number(driver); metadata["cuRAND"] = number(curand);
      std::size_t free, total; cuda_check(cudaMemGetInfo(&free, &total));
      metadata["initial_free_device_bytes"] = number(free); metadata["total_device_bytes"] = number(total);
      char pci[64]{}; cuda_check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
      metadata["device_identifier"] = json_quote(pci);
      std::ostringstream uuid;
      uuid << std::hex << std::setfill('0');
      for (unsigned char byte : prop.uuid.bytes) uuid << std::setw(2) << unsigned(byte);
      metadata["device_uuid"] = json_quote(uuid.str());
      metadata["L2_bytes"] = number(prop.l2CacheSize);
      metadata["shared_memory_per_block"] = number(prop.sharedMemPerBlock);
      metadata["shared_memory_optin_per_block"] = number(prop.sharedMemPerBlockOptin);
      metadata["shared_memory_per_SM"] = number(prop.sharedMemPerMultiprocessor);
      if (o.device_info) { std::cout << json(metadata) << '\n'; return 0; }
      stream = std::make_unique<Stream>();
      Size input_bytes = allocation_bytes(o.config.state_size(), sizeof(cuda::std::complex<double>));
      Size draws = allocation_bytes(std::min(o.config.state_size(), Size{1} << 20), 16);
      preflight_device_bytes(checked_add(checked_add(input_bytes, draws), o.memory_reserve));
      std::cout << "Generating one read-only input; N=" << o.config.state_size() << '\n';
      state = std::make_unique<DeviceState>(generate_state(o.config));
      if (o.validate) {
        std::cout << "Computing full-input chunked CPU references for selected configurations\n";
        refs = references(m, *state, o);
      }
    } catch (const Unsupported& e) { startup = Status::unsupported; startup_reason = e.what(); }
      catch (const std::exception& e) { startup = Status::failure; startup_reason = e.what(); }
    if (o.device_info) throw std::runtime_error("device query failed: " + startup_reason);
    metadata["startup_status"] = json_quote(status_name(startup));
    metadata["startup_reason"] = json_quote(startup_reason);
    Writer writer(o.prefix, metadata);
    std::map<std::string, Size> counts{{"success",0}, {"unsupported",0}, {"failure",0}, {"not_run",0}};
    bool failed = startup == Status::failure;
    bool poisoned = false;
    for (Size id = 0; id < m.cases.size(); ++id) {
      const auto& x = m.cases[id];
      Fields row = row_template(x, id, o);
      // CSV rows carry identical run metadata as JSON's metadata object.
      for (const auto& p : metadata) row["run_" + p.first] = p.second;
      Status status = Status::not_run;
      if (o.profile_case && id != *o.profile_case) row["reason"] = json_quote("not selected for separate profiling run");
      else if (o.run_budget_ms && std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - run_start).count() >= o.run_budget_ms) {
        row["reason"] = json_quote("run budget reached between cases; resume required");
      }
      else if (startup != Status::success) {
        status = startup == Status::unsupported ? Status::unsupported : Status::not_run;
        row["reason"] = json_quote("input/reference setup: " + startup_reason);
        row["timing_status"] = json_quote(status_name(status));
      } else if (poisoned) row["reason"] = json_quote("not attempted after unexpected execution error");
      else try {
        Size workspace = 0;
        if (x.segmented) workspace = segmented_workspace_bytes(x.c);
        else {
          auto r = reduction_resources(x.c, x.storage, x.primitive);
          row["resources"] = json_quote(resource_description(r));
          row["blocks"] = number(r.blocks); kernel(row, r.first, 1);
          kernel(row, r.second, 2);
          workspace = checked_add(allocation_bytes(checked_mul(r.blocks, x.c.bins()), sizeof(double)),
            checked_add(allocation_bytes(x.c.bins(), sizeof(double)), allocation_bytes(x.c.positions.size(), sizeof(unsigned))));
          if (x.storage == AccumulatorStorage::global) workspace = checked_add(workspace,
            allocation_bytes(workspace_elements(r.blocks, x.c.block_size, x.c.bins()), sizeof(double)));
        }
        row["workspace_bytes"] = number(workspace);
        row["total_live_device_bytes"] = number(checked_add(workspace, allocation_bytes(x.c.state_size(), sizeof(cuda::std::complex<double>))));
        row["estimated_workspace_bytes"] = number(workspace);
        if (workspace > o.workspace_limit) throw Unsupported("workspace memory limit exceeded");
        preflight_device_bytes(checked_add(workspace, o.memory_reserve));
        Result result;
        std::vector<double> output;
        auto execute = [&](auto& reduction) {
          if (o.profile_case) {
            ProfileRange range;
            reduction.execute(*state, stream->value); cuda_check(cudaStreamSynchronize(stream->value));
            result.status = Status::success;
            row["profiling_status"] = json_quote("success");
          } else result = reduction.run(*state, output, stream->value);
        };
        if (x.segmented) { SegmentedReduction reduction(x.c); execute(reduction); }
        else { GlobalReduction reduction(x.c, x.layout, x.storage, x.primitive); execute(reduction); }
        status = result.status;
        row["reason"] = json_quote(result.reason);
        if (!o.profile_case) row["timing_status"] = json_quote(status_name(status));
        row["raw_elapsed_ms"] = array(result.raw_elapsed_ms);
        row["completed_repetitions"] = number(result.raw_elapsed_ms.size());
        if (result.raw_elapsed_ms.size() < x.c.repetitions && result.status == Status::success && !o.profile_case)
          row["timing_status"] = json_quote("stopped");
        if (result.elapsed_ms) {
          auto samples = result.raw_elapsed_ms;
          std::sort(samples.begin(), samples.end());
          row["mean_ms"] = number(*result.elapsed_ms);
          row["min_ms"] = number(samples.front()); row["max_ms"] = number(samples.back());
          row["median_ms"] = number((samples[(samples.size()-1)/2] + samples[samples.size()/2]) / 2);
          if (samples.size() > 1) {
            double squares = 0;
            for (double ms : samples) squares += (ms - *result.elapsed_ms) * (ms - *result.elapsed_ms);
            row["sample_stddev_ms"] = number(std::sqrt(squares / (samples.size()-1)));
          }
          if (*result.elapsed_ms > 0) row["input_effective_GB_per_s"] = number(
            double(allocation_bytes(x.c.state_size(), sizeof(cuda::std::complex<double>))) / (*result.elapsed_ms * 1e6));
        }
        if (status == Status::success && o.validate && !o.profile_case) {
          bool ok = true;
          Fields maxima;
          for (const auto& values : result.repetition_outputs) {
            ok &= validate(row, values, refs[x.reference]);
            for (const auto* key : {"max_absolute_error", "max_relative_error", "total_norm_absolute_error"}) {
              if (row[key] != "null" && (maxima[key].empty() || std::stold(row[key]) > std::stold(maxima[key])))
                maxima[key] = row[key];
            }
          }
          for (const auto& p : maxima) row[p.first] = p.second;
          if (!ok) {
            status = Status::failure; row["validation_status"] = json_quote("failure");
            row["validation_reason"] = json_quote("at least one measured repetition failed bin/total norm validation");
          }
        }
        if (result.status == Status::failure) poisoned = true;
      } catch (const Unsupported& e) {
        status = Status::unsupported; row["timing_status"] = json_quote("unsupported"); row["reason"] = json_quote(e.what());
      } catch (const std::exception& e) {
        status = Status::failure; row["reason"] = json_quote(e.what()); poisoned = true;
      }
      row["status"] = json_quote(status_name(status));
      ++counts[status_name(status)]; failed |= status == Status::failure;
      writer.write(row);
      if (o.stream_rows) std::cout << "MEASUREMENT_ROW " << json(row) << std::endl;
      std::cout << "case " << id << ' ' << status_name(status) << ' ' << row["reason"] << '\n';
    }
    Fields summary;
    for (const auto& p : counts) { summary[p.first] = number(p.second); std::cout << p.first << '=' << p.second << ' '; }
    summary["requested_cases"] = number(m.cases.size());
    writer.finish(summary);
    std::cout << "\nResults: " << o.prefix << ".json and " << o.prefix << ".csv\n";
    return failed ? 1 : 0;
  } catch (const std::exception& e) { std::cerr << "failure: " << e.what() << '\n'; return 1; }
}
