#pragma once
#include "measurement/core.hpp"
#include <iterator>
#include <limits>
#ifdef __CUDACC__
#define MEASUREMENT_HD __host__ __device__
#else
#define MEASUREMENT_HD
#endif
namespace measurement {
using LogicalOffset = std::int64_t;
static_assert(sizeof(LogicalOffset) == 8, "64-bit signed iterator arithmetic required");
// Small by-value descriptor; no N-element mapping or norm buffer.
struct GatherMap {
  unsigned qubits = 0, k = 0, positions[max_measurement_qubits]{};
  Size length = 0, measured_mask = 0;
  GatherMap() = default;
  explicit GatherMap(const Config& c) : qubits(c.qubits), k(c.positions.size()) {
    c.validate();
    length = c.state_size() / c.bins();
    for (unsigned t = 0; t < k; ++t) {
      positions[t] = c.positions[t];
      measured_mask |= Size{1} << positions[t];
    }
  }
  MEASUREMENT_HD Size operator()(LogicalOffset v) const {
    const Size bin = Size(v) / length;
    Size remaining = Size(v) % length, j = 0;
    for (unsigned t = 0; t < k; ++t)
      j |= ((bin >> t) & Size{1}) << positions[t];
    for (unsigned bit = 0; bit < qubits; ++bit)
      if (!(measured_mask & (Size{1} << bit))) {
        j |= (remaining & Size{1}) << bit;
        remaining >>= 1;
      }
    return j;
  }
};
struct SegmentOffset {
  LogicalOffset length = 0;
  MEASUREMENT_HD LogicalOffset operator()(LogicalOffset bin) const { return bin * length; }
};
// A lazy value-returning random-access iterator. Index and difference never use int.
template<class Function, class Value>
struct LogicalIterator {
  using value_type = Value;
  using difference_type = LogicalOffset;
  using reference = Value;
  using pointer = void;
  using iterator_category = std::random_access_iterator_tag;
  Function function;
  LogicalOffset index = 0;
  MEASUREMENT_HD Value operator*() const { return function(index); }
  MEASUREMENT_HD Value operator[](difference_type n) const { return function(index + n); }
  MEASUREMENT_HD LogicalIterator& operator+=(difference_type n) { index += n; return *this; }
  MEASUREMENT_HD LogicalIterator& operator-=(difference_type n) { index -= n; return *this; }
  MEASUREMENT_HD LogicalIterator operator+(difference_type n) const { auto r = *this; return r += n; }
  MEASUREMENT_HD LogicalIterator operator-(difference_type n) const { auto r = *this; return r -= n; }
  MEASUREMENT_HD friend LogicalIterator operator+(difference_type n, LogicalIterator r) { return r += n; }
  MEASUREMENT_HD difference_type operator-(LogicalIterator r) const { return index - r.index; }
  MEASUREMENT_HD LogicalIterator& operator++() { return *this += 1; }
  MEASUREMENT_HD LogicalIterator operator++(int) { auto r = *this; ++*this; return r; }
  MEASUREMENT_HD LogicalIterator& operator--() { return *this -= 1; }
  MEASUREMENT_HD LogicalIterator operator--(int) { auto r = *this; --*this; return r; }
  MEASUREMENT_HD bool operator==(LogicalIterator r) const { return index == r.index; }
  MEASUREMENT_HD bool operator!=(LogicalIterator r) const { return index != r.index; }
  MEASUREMENT_HD bool operator<(LogicalIterator r) const { return index < r.index; }
  MEASUREMENT_HD bool operator>(LogicalIterator r) const { return index > r.index; }
  MEASUREMENT_HD bool operator<=(LogicalIterator r) const { return index <= r.index; }
  MEASUREMENT_HD bool operator>=(LogicalIterator r) const { return index >= r.index; }
};
using OffsetIterator = LogicalIterator<SegmentOffset, LogicalOffset>;
} // namespace measurement
#undef MEASUREMENT_HD
