// sbmc_trace: tool of the tests, not an application of the project. It runs the sequential
// reference C0 (run_sb_sequential) and writes a binary trace file read by T8
// (tests/python/test_t8.py): the dense J, the parameters, p[m], x0, the states after the first K
// steps and the results of every replica.
//
// Usage: sbmc_trace (--rudy FILE | --dense N --instance-seed SEED) --algorithm bsb|dsb
//                   --replicas R --steps M --record K --seed S --out FILE
//                   [--dt X] [--a0 X] [--c0 X|auto] [--eval-every K] [--first-replica F]
// Every option is "--name value" (two arguments). Integers are decimal and must fit their type;
// floats are read with std::from_chars and must be finite; 1 <= K <= M. Defaults: dt 1.25, a0 1,
// c0 auto, eval-every 0, first-replica 0. The other limits (R >= 1, c0 >= 0, ...) are those of
// run_sb_sequential.
//
// Exit codes: 0 success (no output); 1 error of the library, of the instance (also n * n > 2^24
// and, for a rudy graph, |w| > 127) or of the output file; 2 invalid command line.
//
// Trace format: little-endian, contiguous fields, no padding.
//   offset 0   char[8]          "SBMCTRC1"
//   offset 8   int64            n
//   offset 16  int32            R
//   offset 20  int32            M
//   offset 24  int32            K
//   offset 28  int32            algorithm: 0 bSB, 1 dSB
//   offset 32  int32            eval_every
//   offset 36  float32          dt
//   offset 40  float32          a0
//   offset 44  float32          c0 used
//   offset 48  uint64           run seed
//   offset 56  int64            first_replica
//   offset 64  int64            W
//   offset 72  float32[M]       p[m]
//   then       int8[n * n]      J, row-major, zero diagonal
//   then       float32[R * n]   x0, replica-major
//   then       float32[K * R * n]  trace_x, [k][r][i]: state after step k + 1
//   then       float32[K * R * n]  trace_y
//   then       int64[R]         best_energy
//   then       int64[R]         best_cut
//   then       int32[R]         best_step
//   then       int64[R]         final_cut
// Total size: 72 + 4M + n^2 + 4Rn + 8KRn + 28R bytes.
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <sbmc/initial_state.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sb.hpp>
#include <sbmc/sb_sequential.hpp>

static_assert(std::endian::native == std::endian::little, "the trace format is little-endian");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the trace stores IEEE-754 binary32 floats");

namespace {

constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
constexpr std::int64_t kMaxSpins = 4096;  // n * n <= 2^24: the trace contains the dense J
constexpr std::string_view kUsage =
    "usage: sbmc_trace (--rudy FILE | --dense N --instance-seed SEED) --algorithm bsb|dsb\n"
    "                  --replicas R --steps M --record K --seed S --out FILE\n"
    "                  [--dt X] [--a0 X] [--c0 X|auto] [--eval-every K] [--first-replica F]\n";

class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Options {
  std::optional<std::string> rudy;
  std::optional<std::int64_t> dense;
  std::optional<std::uint64_t> instance_seed;
  sbmc::SbParams params;
  int record = 0;
  std::string out;
};

template <typename T>
T parse_integer(const std::string& name, const std::string& text) {
  T value{};
  const char* const end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (text.empty() || ec != std::errc{} || ptr != end) {
    throw UsageError(name + ": '" + text + "' is not a decimal integer of the allowed range");
  }
  return value;
}

float parse_float(const std::string& name, const std::string& text) {
  float value = 0.0f;
  const char* const end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (text.empty() || ec != std::errc{} || ptr != end || !std::isfinite(value)) {
    throw UsageError(name + ": '" + text + "' is not a finite number");
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  static const char* const kKnown[] = {
      "--rudy",  "--dense",  "--instance-seed", "--algorithm",    "--replicas",
      "--steps", "--record", "--seed",          "--out",          "--dt",
      "--a0",    "--c0",     "--eval-every",    "--first-replica"};
  std::map<std::string, std::string> values;
  for (int k = 1; k < argc; k += 2) {
    const std::string name = argv[k];
    bool known = false;
    for (const char* option : kKnown) {
      known = known || name == option;
    }
    if (!known) {
      throw UsageError("unknown option '" + name + "'");
    }
    if (k + 1 >= argc) {
      throw UsageError(name + " needs a value");
    }
    if (!values.emplace(name, argv[k + 1]).second) {
      throw UsageError(name + " is given more than once");
    }
  }
  const auto required = [&](const char* name) -> const std::string& {
    const auto it = values.find(name);
    if (it == values.end()) {
      throw UsageError(std::string("missing option ") + name);
    }
    return it->second;
  };

  Options o;
  const bool has_rudy = values.count("--rudy") != 0;
  const bool has_dense = values.count("--dense") != 0;
  const bool has_instance_seed = values.count("--instance-seed") != 0;
  if (has_rudy == has_dense) {
    throw UsageError("exactly one instance is needed: --rudy FILE or --dense N");
  }
  if (has_dense != has_instance_seed) {
    throw UsageError("--instance-seed goes with --dense, and --dense needs it");
  }
  if (has_rudy) {
    o.rudy = values["--rudy"];
  } else {
    o.dense = parse_integer<std::int64_t>("--dense", values["--dense"]);
    o.instance_seed = parse_integer<std::uint64_t>("--instance-seed", values["--instance-seed"]);
  }

  const std::string& algorithm = required("--algorithm");
  if (algorithm == "bsb") {
    o.params.algorithm = sbmc::SbAlgorithm::ballistic;
  } else if (algorithm == "dsb") {
    o.params.algorithm = sbmc::SbAlgorithm::discrete;
  } else {
    throw UsageError("--algorithm: '" + algorithm + "' is neither bsb nor dsb");
  }
  o.params.replicas = parse_integer<int>("--replicas", required("--replicas"));
  o.params.steps = parse_integer<int>("--steps", required("--steps"));
  o.record = parse_integer<int>("--record", required("--record"));
  o.params.seed = parse_integer<std::uint64_t>("--seed", required("--seed"));
  o.out = required("--out");
  if (o.record < 1 || o.record > o.params.steps) {
    throw UsageError("--record " + std::to_string(o.record) + " is outside [1, --steps " +
                     std::to_string(o.params.steps) + "]");
  }

  o.params.dt = values.count("--dt") ? parse_float("--dt", values["--dt"]) : 1.25f;
  o.params.a0 = values.count("--a0") ? parse_float("--a0", values["--a0"]) : 1.0f;
  if (values.count("--c0") && values["--c0"] != "auto") {
    o.params.c0 = parse_float("--c0", values["--c0"]);
  } else {
    o.params.c0 = 0.0f;  // automatic
  }
  o.params.eval_every =
      values.count("--eval-every") ? parse_integer<int>("--eval-every", values["--eval-every"]) : 0;
  o.params.first_replica =
      values.count("--first-replica")
          ? parse_integer<std::int64_t>("--first-replica", values["--first-replica"])
          : 0;
  return o;
}

template <typename T>
void put(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void put_all(std::ofstream& out, const std::vector<T>& values) {
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void write_trace(const Options& o, std::int64_t n, const std::vector<std::int8_t>& j,
                 const std::vector<float>& p, const std::vector<float>& x0,
                 const sbmc::SbResult& r) {
  std::ofstream out(o.out, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("cannot open '" + o.out + "' for writing");
  }
  out.write("SBMCTRC1", 8);
  put(out, n);
  put(out, std::int32_t{r.replicas});
  put(out, std::int32_t{r.steps});
  put(out, std::int32_t{r.trace_steps});
  put(out, std::int32_t{o.params.algorithm == sbmc::SbAlgorithm::discrete ? 1 : 0});
  put(out, std::int32_t{o.params.eval_every});
  put(out, o.params.dt);
  put(out, o.params.a0);
  put(out, r.c0);
  put(out, o.params.seed);
  put(out, r.first_replica);
  put(out, r.total_weight);
  put_all(out, p);
  put_all(out, j);
  put_all(out, x0);
  put_all(out, r.trace_x);
  put_all(out, r.trace_y);
  for (const auto& rep : r.per_replica) {
    put(out, rep.best_energy);
  }
  for (const auto& rep : r.per_replica) {
    put(out, rep.best_cut);
  }
  for (const auto& rep : r.per_replica) {
    put(out, std::int32_t{rep.best_step});
  }
  for (const auto& rep : r.per_replica) {
    put(out, rep.final_cut);
  }

  const std::int64_t replicas = r.replicas;
  const std::int64_t expected = 72 + 4 * std::int64_t{r.steps} + n * n + 4 * replicas * n +
                                8 * std::int64_t{r.trace_steps} * replicas * n + 28 * replicas;
  const std::int64_t written = out.tellp();
  out.close();
  if (out.fail()) {
    throw std::runtime_error("cannot write '" + o.out + "'");
  }
  if (written != expected) {
    throw std::runtime_error("internal error: " + std::to_string(written) +
                             " bytes written, the format needs " + std::to_string(expected));
  }
}

int run(const Options& o) {
  const sbmc::Instance inst =
      o.rudy ? sbmc::Instance{sbmc::read_rudy(*o.rudy)}
             : sbmc::Instance{sbmc::DenseSpinGlass{*o.dense, *o.instance_seed}};
  const std::int64_t n = sbmc::num_spins(inst);
  if (n > kMaxSpins) {
    throw std::runtime_error("n = " + std::to_string(n) +
                             ": the trace contains the dense J and needs n * n <= 2^24");
  }
  std::vector<std::int8_t> j(static_cast<std::size_t>(n * n));
  sbmc::dense_rows_int8(inst, 0, n, n, j);  // for a graph: |w| <= 127
  const sbmc::SbResult result = sbmc::run_sb_sequential(inst, o.params, o.record);
  const std::vector<float> p = sbmc::pressure_schedule(o.params.a0, o.params.steps);
  const std::vector<float> x0 =
      sbmc::make_initial_state(n, o.params.replicas, o.params.seed, o.params.first_replica).x0;
  write_trace(o, n, j, p, x0, result);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const UsageError& e) {
    std::cerr << "sbmc_trace: " << e.what() << '\n' << kUsage;
    return kExitUsage;
  }
  try {
    return run(options);
  } catch (const std::exception& e) {
    std::cerr << "sbmc_trace: " << e.what() << '\n';
    return kExitError;
  }
}
