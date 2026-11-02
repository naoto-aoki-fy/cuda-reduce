#include "measurement/core.hpp"
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace measurement {
Size checked_add(Size a, Size b) {
  if (b > std::numeric_limits<Size>::max() - a) throw std::overflow_error("size addition overflow");
  return a + b;
}
Size checked_mul(Size a, Size b) {
  if (a && b > std::numeric_limits<Size>::max() / a) throw std::overflow_error("size multiplication overflow");
  return a * b;
}
Size power_of_two(unsigned exponent) {
  if (exponent >= 64) throw std::invalid_argument("power exponent must be below 64");
  return Size{1} << exponent;
}
std::size_t allocation_bytes(Size elements, Size element_bytes) {
  Size bytes = checked_mul(elements, element_bytes);
  if (bytes > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("allocation exceeds size_t");
  return static_cast<std::size_t>(bytes);
}
void Config::validate() const {
  if (qubits == 0 || qubits >= 64) throw std::invalid_argument("qubits must be in [1,63]");
  if (positions.empty() || positions.size() > max_measurement_qubits || positions.size() > qubits)
    throw std::invalid_argument("measurement count must be in [1,min(16,q)]");
  Size seen = 0;
  for (unsigned p : positions) {
    if (p >= qubits) throw std::invalid_argument("position outside state index");
    Size bit = Size{1} << p;
    if (seen & bit) throw std::invalid_argument("duplicate position");
    seen |= bit;
  }
  if (block_size != 32 && block_size != 64 && block_size != 128 &&
      block_size != 256 && block_size != 512 && block_size != 1024)
    throw std::invalid_argument("supported block sizes: 32,64,128,256,512,1024");
  if (!sm_multiplier || !repetitions || !chunk_elements)
    throw std::invalid_argument("multiplier, repetitions and chunk size must be positive");
  allocation_bytes(state_size(), 2 * sizeof(double));
  allocation_bytes(chunk_elements, sizeof(std::complex<double>));
}
Size Config::state_size() const { return power_of_two(qubits); }
Size Config::bins() const { return power_of_two(static_cast<unsigned>(positions.size())); }
static Size number(std::string_view s) {
  Size value{};
  auto result = std::from_chars(s.data(), s.data() + s.size(), value);
  if (s.empty() || result.ec != std::errc{} || result.ptr != s.data() + s.size())
    throw std::invalid_argument("invalid unsigned integer: " + std::string(s));
  return value;
}
static unsigned small_number(std::string_view s) {
  Size n = number(s);
  if (n > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("integer exceeds unsigned range");
  return static_cast<unsigned>(n);
}
Config parse_config(int argc, const char* const* argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    std::string_view key(argv[i]);
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    std::string_view value(argv[++i]);
    if (key == "--qubits") c.qubits = small_number(value);
    else if (key == "--positions") {
      c.positions.clear();
      for (;;) {
        auto comma = value.find(',');
        c.positions.push_back(small_number(value.substr(0, comma)));
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
      }
    } else if (key == "--block-size") c.block_size = small_number(value);
    else if (key == "--sm-multiplier") c.sm_multiplier = number(value);
    else if (key == "--repetitions") c.repetitions = number(value);
    else if (key == "--warmups") c.warmups = number(value);
    else if (key == "--seed") c.seed = number(value);
    else if (key == "--case-budget-ms") c.case_budget_ms = number(value);
    else if (key == "--chunk-elements") c.chunk_elements = number(value);
    else throw std::invalid_argument("unknown option: " + std::string(key));
  }
  c.validate();
  return c;
}
std::vector<unsigned> contiguous_positions(unsigned q, unsigned k, const std::string& band) {
  if (!q || q >= 64 || !k || k > max_measurement_qubits || k > q)
    throw std::invalid_argument("invalid q or k");
  unsigned start;
  if (band == "low") start = 0;
  else if (band == "middle") start = (q-k)/2;
  else if (band == "high") start = q-k;
  else throw std::invalid_argument("unknown position band");
  std::vector<unsigned> result;
  for (unsigned t = 0; t < k; ++t) result.push_back(start+t);
  return result;
}
Size launch_blocks(const Config& c, Size sm_count) {
  c.validate();
  if (!sm_count) throw std::invalid_argument("zero SM count");
  Size blocks = checked_mul(sm_count, c.sm_multiplier);
  if (blocks > static_cast<Size>(std::numeric_limits<int>::max()))
    throw std::overflow_error("grid x exceeds CUDA launch limit");
  return blocks;
}
Size workspace_elements(Size blocks, Size threads, Size bins) {
  return checked_mul(checked_mul(blocks, threads), bins);
}
Size bin_for(Size index, const std::vector<unsigned>& positions) {
  if (positions.empty() || positions.size() > max_measurement_qubits)
    throw std::invalid_argument("invalid position count");
  Size bin = 0, seen = 0;
  for (std::size_t t = 0; t < positions.size(); ++t) {
    unsigned p = positions[t];
    if (p >= 64) throw std::invalid_argument("invalid shift");
    Size mask = Size{1} << p;
    if (seen & mask) throw std::invalid_argument("duplicate position");
    seen |= mask;
    bin |= ((index >> p) & Size{1}) << t;
  }
  return bin;
}
CpuReference::CpuReference(const Config& c) : config_(c) {
  c.validate();
  sums_.resize(c.bins()); corrections_.resize(c.bins());
}
void CpuReference::add_chunk(Size offset, const std::complex<double>* values, Size count) {
  Size end = checked_add(offset, count);
  if (offset != next_ || end > config_.state_size() || (count && !values))
    throw std::invalid_argument("invalid reference chunk");
  for (Size i = 0; i < count; ++i) {
    long double re = values[i].real(), im = values[i].imag();
    long double x = re*re + im*im;
    auto b = bin_for(offset+i, config_.positions);
    long double sum = sums_[b], total = sum + x;
    corrections_[b] += std::abs(sum) >= std::abs(x) ? (sum-total)+x : (x-total)+sum;
    sums_[b] = total;
  }
  next_ = end;
}
std::vector<long double> CpuReference::finish() const {
  if (next_ != config_.state_size()) throw std::logic_error("reference input incomplete");
  auto result = sums_;
  for (std::size_t i = 0; i < result.size(); ++i) result[i] += corrections_[i];
  return result;
}
std::vector<Fixture> deterministic_fixtures() {
  Config config;
  config.qubits = 3;
  config.positions = {2, 0};
  std::vector<std::complex<double>> zeros(8);
  auto single = zeros;
  single[5] = {3, 4};
  auto multiple = zeros;
  multiple[0] = {1, 0};
  multiple[1] = {0, 2};
  multiple[2] = {3, 0};
  multiple[4] = {3, 4};
  multiple[5] = {2, 3};
  multiple[7] = {1, 1};
  Config reversed = config;
  reversed.positions = {0, 2};
  return {
    {"all zeros", config, zeros, {0, 0, 0, 0}},
    {"single nonzero", config, single, {0, 0, 0, 25}},
    {"noncontiguous positions", config, multiple, {10, 25, 4, 15}},
    {"reversed positions", reversed, multiple, {10, 4, 25, 15}}
  };
}
const char* status_name(Status s) {
  switch (s) {
    case Status::success: return "success";
    case Status::unsupported: return "unsupported";
    case Status::failure: return "failure";
    case Status::not_run: return "not_run";
  }
  throw std::invalid_argument("unknown result status");
}
} // namespace measurement
