// sbmc_sa_thresholds: tool of the tests, not an application of the project (phase 07, part 4,
// section 5.4 of F07-C4). The comparison program outside the library for the bitwise equality of
// simulated annealing between the Windows build (MinGW-w64, its libm) and the Linux build (glibc):
// it writes the schedule and the acceptance thresholds of the library, sbmc::sa_beta and
// sbmc::sa_threshold (the only floating-point operations of SA: std::pow and std::exp in double),
// so that python/sbmc_e9.py bitwise can compare the two platforms line by line.
//
// Usage: sbmc_sa_thresholds --beta0 X --beta1 X --sweeps S --max-delta D
// Every option is "--name value". Output, one line per sweep k = 0 .. S - 1:
//   <k> <beta_k> <t(1)>,<t(2)>,...,<t(d)>
// with beta_k in the shortest form that reads back as the same double and t(e) =
// sa_threshold(beta_k, e) for e = 1 .. d, where d is D or the first e with t(e) = 0 (the
// thresholds do not increase with e), whichever comes first.
//
// Exit codes: 0 success; 1 error of the library (beta0 > beta1, ...); 2 invalid command line.
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include <sbmc/sa.hpp>

namespace {

template <typename T>
std::optional<T> parse(std::string_view text) {
  T value{};
  const char* const last = text.data() + text.size();
  const auto [end, error] = std::from_chars(text.data(), last, value);
  if (text.empty() || error != std::errc{} || end != last) {
    return std::nullopt;
  }
  return value;
}

int usage(std::string_view message) {
  std::cerr << "sbmc_sa_thresholds: " << message
            << "\nusage: sbmc_sa_thresholds --beta0 X --beta1 X --sweeps S --max-delta D\n";
  return 2;
}

std::string shortest(double value) {
  std::array<char, 32> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("?");
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<double> beta0;
  std::optional<double> beta1;
  std::optional<int> sweeps;
  std::optional<std::int64_t> max_delta;
  for (int k = 1; k < argc; k += 2) {
    if (k + 1 >= argc) {
      return usage(std::string("missing value of ") + argv[k]);
    }
    const std::string_view name = argv[k];
    const std::string_view value = argv[k + 1];
    if (name == "--beta0") {
      beta0 = parse<double>(value);
    } else if (name == "--beta1") {
      beta1 = parse<double>(value);
    } else if (name == "--sweeps") {
      sweeps = parse<int>(value);
    } else if (name == "--max-delta") {
      max_delta = parse<std::int64_t>(value);
    } else {
      return usage("unknown option " + std::string(name));
    }
  }
  if (!beta0 || !beta1 || !sweeps || !max_delta || *sweeps < 1 || *max_delta < 1) {
    return usage("--beta0, --beta1, --sweeps >= 1 and --max-delta >= 1 are required numbers");
  }
  try {
    std::string line;
    for (int k = 0; k < *sweeps; ++k) {
      const double beta = sbmc::sa_beta(*beta0, *beta1, *sweeps, k);
      line = std::to_string(k) + " " + shortest(beta) + " ";
      for (std::int64_t e = 1; e <= *max_delta; ++e) {
        const std::uint64_t threshold = sbmc::sa_threshold(beta, e);
        line += (e > 1 ? "," : "") + std::to_string(threshold);
        if (threshold == 0) {
          break;
        }
      }
      line += '\n';
      std::fwrite(line.data(), 1, line.size(), stdout);
    }
    std::fflush(stdout);
  } catch (const std::exception& e) {
    std::cerr << "sbmc_sa_thresholds: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
