// Simulated annealing, parts common to every variant (sbmc/sa.hpp): the schedule of beta and the
// acceptance threshold. One definition, compiled with the options of the library, called by the
// sequential reference and by the optimized variants, so that every variant gets the same bits.
#include <sbmc/sa.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <system_error>

#include <sbmc/error.hpp>

namespace sbmc {
namespace {

// Shortest form that reads back as the same double ("inf", "nan" included).
std::string number(double v) {
  std::array<char, 32> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), v);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("?");
}

}  // namespace

double sa_beta(double beta0, double beta1, int sweeps, int k) {
  if (sweeps < 1) {
    throw InvalidArgument("sa_beta: sweeps = " + std::to_string(sweeps) + " must be at least 1");
  }
  if (k < 0 || k >= sweeps) {
    throw InvalidArgument("sa_beta: k = " + std::to_string(k) +
                          " is outside [0, sweeps = " + std::to_string(sweeps) + ")");
  }
  if (!std::isfinite(beta0) || !(beta0 > 0.0)) {
    throw InvalidArgument("sa_beta: beta0 = " + number(beta0) + " must be finite and > 0");
  }
  if (!std::isfinite(beta1) || !(beta1 >= beta0)) {
    throw InvalidArgument("sa_beta: beta1 = " + number(beta1) +
                          " must be finite and >= beta0 = " + number(beta0));
  }
  const double t = (sweeps > 1) ? double(k) / double(sweeps - 1) : 0.0;
  return beta0 * std::pow(beta1 / beta0, t);
}

std::uint64_t sa_threshold(double beta, std::int64_t delta_e) {
  if (!std::isfinite(beta) || !(beta > 0.0)) {
    throw InvalidArgument("sa_threshold: beta = " + number(beta) + " must be finite and > 0");
  }
  if (delta_e <= 0) {
    throw InvalidArgument("sa_threshold: delta_e = " + std::to_string(delta_e) +
                          " must be > 0 (downhill and neutral moves are always accepted)");
  }
  const double p = std::exp(-(beta * static_cast<double>(delta_e)));
  return static_cast<std::uint64_t>(p * 4294967296.0);
}

}  // namespace sbmc
