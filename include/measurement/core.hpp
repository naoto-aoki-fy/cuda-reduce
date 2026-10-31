#pragma once
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace measurement {
using Size = std::uint64_t;
constexpr Size default_seed = 20261031;
constexpr unsigned max_measurement_qubits = 16;
Size checked_add(Size a, Size b);
Size checked_mul(Size a, Size b);
Size power_of_two(unsigned exponent);
std::size_t allocation_bytes(Size elements, Size element_bytes);
struct Config {
  unsigned qubits = 10;
  std::vector<unsigned> positions{0};
  unsigned block_size = 256;
  Size sm_multiplier = 8;
  Size repetitions = 1;
  Size warmups = 3;
  Size seed = default_seed;
  Size chunk_elements = 65536;
  void validate() const;
  Size state_size() const;
  Size bins() const;
};
Config parse_config(int argc, const char* const* argv);
std::vector<unsigned> contiguous_positions(unsigned q, unsigned k, const std::string& band);
Size launch_blocks(const Config& config, Size sm_count);
Size workspace_elements(Size blocks, Size threads, Size bins);
Size bin_for(Size index, const std::vector<unsigned>& positions);
class CpuReference {
 public:
  explicit CpuReference(const Config& config);
  // Chunks must cover the state in order, without gaps or overlap.
  void add_chunk(Size offset, const std::complex<double>* values, Size count);
  std::vector<long double> finish() const;
 private:
  Config config_;
  Size next_ = 0;
  std::vector<long double> sums_, corrections_;
};
// Small, deterministic inputs shared by future backend tests.
struct Fixture {
  std::string name;
  Config config;
  std::vector<std::complex<double>> amplitudes;
  std::vector<long double> expected;
};
std::vector<Fixture> deterministic_fixtures();
enum class Status { success, unsupported, failure, not_run };
const char* status_name(Status status);
struct Result {
  Status status = Status::not_run;
  std::string reason;
  std::vector<double> raw_elapsed_ms;
  std::vector<std::vector<double>> repetition_outputs;
  std::optional<double> elapsed_ms;
  std::optional<double> dram_bytes; // nullopt means unavailable, never an invented zero.
};
} // namespace measurement
