// Logical processors, efficiency classes and the process masks of sbmc_run --affinity (phase 07,
// part 4, E9): the rules on a list of logical processors, without any API of the operating
// system, so that they are compiled and tested on every platform (tests/cpu/test_cpu_topology.cpp).
// The list comes from apps/win_platform.cpp on Windows, the only file with Windows API.
//
// A logical processor has its index (the bit of the affinity mask, group 0 only), its physical
// core and its efficiency class (higher: faster cores). The masks of --affinity:
//   p   the logical processors of the highest class (P cores);
//   e   those of the lowest class (E cores);
//   p1  for every core of the highest class, its logical processor of smallest index (one thread
//       per P core);
//   all every logical processor.
// With one class p and p1 are all, and e is an error. More than 64 logical processors are an error
// (one mask of 64 bits).
//
// Text forms: cpulist, the format of Cpus_allowed_list of Linux (increasing indices, runs of two
// or more as a-b: "0-11", "0,2,4", "0-3,8"); cpu classes, groups "<class>:<cpulist>" of the
// processors of a mask, separated by a space, in decreasing class ("1:0-11 0:12-15").
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sbmc_app {

struct LogicalProcessor {
  int index = 0;             // bit of the affinity mask, 0..63
  int core = 0;              // physical core
  int efficiency_class = 0;  // higher: faster
};

enum class Affinity { p1, p, e, all };

inline constexpr int kMaxLogicalProcessors = 64;

// "p1", "p", "e" or "all"; nullopt otherwise.
inline std::optional<Affinity> parse_affinity(std::string_view text) {
  if (text == "p1") {
    return Affinity::p1;
  }
  if (text == "p") {
    return Affinity::p;
  }
  if (text == "e") {
    return Affinity::e;
  }
  if (text == "all") {
    return Affinity::all;
  }
  return std::nullopt;
}

inline std::string affinity_name(Affinity affinity) {
  switch (affinity) {
    case Affinity::p1:
      return "p1";
    case Affinity::p:
      return "p";
    case Affinity::e:
      return "e";
    case Affinity::all:
      break;
  }
  return "all";
}

// The checks of a list: not empty, at most 64 processors, indices in 0..63 and distinct.
inline void check_processors(std::span<const LogicalProcessor> processors) {
  if (processors.empty()) {
    throw std::runtime_error("no logical processors");
  }
  if (processors.size() > static_cast<std::size_t>(kMaxLogicalProcessors)) {
    throw std::runtime_error(std::to_string(processors.size()) +
                             " logical processors: more than 64 are not supported (one mask "
                             "of 64 bits)");
  }
  std::uint64_t seen = 0;
  for (const LogicalProcessor& p : processors) {
    if (p.index < 0 || p.index >= kMaxLogicalProcessors) {
      throw std::runtime_error("logical processor " + std::to_string(p.index) +
                               ": index outside 0..63 (more than 64 logical processors or more "
                               "than one processor group are not supported)");
    }
    const std::uint64_t bit = std::uint64_t{1} << p.index;
    if ((seen & bit) != 0) {
      throw std::runtime_error("logical processor " + std::to_string(p.index) + " listed twice");
    }
    seen |= bit;
  }
}

// Number of distinct efficiency classes.
inline int class_count(std::span<const LogicalProcessor> processors) {
  std::vector<int> classes;
  for (const LogicalProcessor& p : processors) {
    classes.push_back(p.efficiency_class);
  }
  std::sort(classes.begin(), classes.end());
  return static_cast<int>(std::unique(classes.begin(), classes.end()) - classes.begin());
}

// The mask of an --affinity; std::invalid_argument for e with one class.
inline std::uint64_t affinity_mask(std::span<const LogicalProcessor> processors,
                                   Affinity affinity) {
  check_processors(processors);
  const auto [low, high] =
      std::minmax_element(processors.begin(), processors.end(),
                          [](const LogicalProcessor& a, const LogicalProcessor& b) {
                            return a.efficiency_class < b.efficiency_class;
                          });
  const int lowest = low->efficiency_class;
  const int highest = high->efficiency_class;
  const bool one_class = lowest == highest;
  if (affinity == Affinity::e && one_class) {
    throw std::invalid_argument(
        "--affinity e: the processor has one efficiency class (no E "
        "cores)");
  }
  std::uint64_t mask = 0;
  if (affinity == Affinity::all || (one_class && affinity != Affinity::e)) {
    for (const LogicalProcessor& p : processors) {
      mask |= std::uint64_t{1} << p.index;
    }
    return mask;
  }
  if (affinity == Affinity::p || affinity == Affinity::e) {
    const int wanted = affinity == Affinity::p ? highest : lowest;
    for (const LogicalProcessor& p : processors) {
      if (p.efficiency_class == wanted) {
        mask |= std::uint64_t{1} << p.index;
      }
    }
    return mask;
  }
  // p1: per core of the highest class, its smallest index.
  std::map<int, int> first_of_core;
  for (const LogicalProcessor& p : processors) {
    if (p.efficiency_class != highest) {
      continue;
    }
    const auto [it, inserted] = first_of_core.emplace(p.core, p.index);
    if (!inserted) {
      it->second = std::min(it->second, p.index);
    }
  }
  for (const auto& [core, index] : first_of_core) {
    mask |= std::uint64_t{1} << index;
  }
  return mask;
}

// cpulist of a mask: "0-11", "0,2,4", "0-3,8"; empty for 0.
inline std::string cpulist(std::uint64_t mask) {
  std::string text;
  int k = 0;
  while (k < kMaxLogicalProcessors) {
    if (((mask >> k) & 1U) == 0) {
      ++k;
      continue;
    }
    int last = k;
    while (last + 1 < kMaxLogicalProcessors && ((mask >> (last + 1)) & 1U) != 0) {
      ++last;
    }
    if (!text.empty()) {
      text += ',';
    }
    text += std::to_string(k);
    if (last > k) {
      text += '-';
      text += std::to_string(last);
    }
    k = last + 1;
  }
  return text;
}

// The classes of the processors of a mask: "1:0-11 0:12-15" (decreasing class); processors of
// the mask that are not in the list are ignored; empty when none is.
inline std::string cpu_classes(std::span<const LogicalProcessor> processors, std::uint64_t mask) {
  std::map<int, std::uint64_t, std::greater<>> by_class;
  for (const LogicalProcessor& p : processors) {
    if (p.index >= 0 && p.index < kMaxLogicalProcessors && ((mask >> p.index) & 1U) != 0) {
      by_class[p.efficiency_class] |= std::uint64_t{1} << p.index;
    }
  }
  std::string text;
  for (const auto& [efficiency_class, class_mask] : by_class) {
    if (!text.empty()) {
      text += ' ';
    }
    text += std::to_string(efficiency_class) + ":" + cpulist(class_mask);
  }
  return text;
}

inline int mask_count(std::uint64_t mask) {
  return std::popcount(mask);
}

}  // namespace sbmc_app
