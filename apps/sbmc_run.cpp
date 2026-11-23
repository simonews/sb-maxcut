// sbmc_run: runs a solver of the project on one instance and writes one CSV row per repetition,
// with the results schema 2 (sbmc/results.hpp). Simulated bifurcation, dSB or bSB: phase 04, the
// sequential reference C0 (variant c0); phase 05, C1 with OpenMP over blocks of replicas
// (variant c1, --threads). Simulated annealing (--algo sa), phase 06: the sequential reference
// (variant c0) and C2 with OpenMP over blocks of replicas (variant c2, --threads). Phase 07: the
// automatic dt and c0 from the estimate of the spectrum of J (sbmc/spectrum.hpp); part 4, the
// build for Windows (E9): --affinity and --print-topology, Windows only (apps/win_platform.cpp,
// the only file with Windows API), and --version on every platform.
//
//   sbmc_run (--instance FILE | --dense N --instance-seed S) --algo dsb|bsb|sa --replicas R
//            [--steps M --dt auto|X [--dt-factor F] | --sweeps S --beta0 X --beta1 X]
//            [--name NAME] [--variant c0|c1|c2] [--threads T] [--a0 X] [--c0 auto|lambda|X]
//            [--eval-every K]
//            [--seed S] [--reps K] [--warmup K] [--target best-known|none|INTEGER]
//            [--best-known FILE] [--out FILE] [--replica-out FILE] [--notes TEXT] [--tag TEXT]
//            [--affinity p1|p|e|all]
//   sbmc_run --print-topology [--affinity p1|p|e|all]
//   sbmc_run --version
//
// --steps and --dt are required with dsb and bsb, which take the variants c0 and c1; --sweeps,
// --beta0 and --beta1 with sa, which takes the variants c0 and c2 and where --steps, --dt, --c0,
// --a0 and --eval-every are not used. --threads only with c1 or c2. --replica-out needs --out and
// another file (compared with std::filesystem::weakly_canonical). dt of SB: --dt auto, dt =
// F * 1.25 * sqrt(2 / (1 - lambda_min / lambda_max)) with F = --dt-factor (default 1, allowed only
// with --dt auto), or a finite number > 0. c0 of SB: --c0 auto, 1 / (2 sigma sqrt(n)) (auto_c0);
// --c0 lambda, 1 / lambda_max; or a finite number > 0. lambda_min and lambda_max come from
// estimate_spectrum with kSpectrumIterations steps, run once per invocation and only for --dt auto
// or --c0 lambda.
//
// Windows (phase 07, part 4): --affinity applies the process mask of the class rules of
// apps/cpu_topology.hpp (p1: one logical processor per P core; p: the P cores; e: the E cores;
// all) with SetProcessAffinityMask, right after the command line is read and before any call of
// OpenMP, so that the threads of the solver run free inside the mask; e on a processor with one
// efficiency class is an error of the command line. --print-topology writes the logical
// processors (index, core, class), the classes, the masks of the four rules, the effective mask
// and omp_get_num_procs, omp_get_max_threads and omp_get_proc_bind, after --affinity when given,
// and exits with 0. On the other platforms both options are errors of the command line (exit
// status 2). --version writes "sbmc_run git_hash=<hash> dirty=<0|1|-1> compiler=<compiler>
// build_type=<type>" and exits with 0.
//
// Run: at startup, before any run, the environment of the process is read once: the logical
// processors allowed to it (the Cpus_allowed_list line of /proc/self/status, column cpu_affinity;
// on Windows the cpulist of the process mask, and the column cpu_classes, the efficiency classes
// of its processors) and the variables OMP_PLACES and OMP_PROC_BIND (columns omp_places and
// omp_proc_bind; empty when missing); host and os (on Windows the computer name and "Windows
// <major>.<minor>.<build>"). Then the instance is read and, with --dt auto or --c0 lambda, the
// spectrum of J is
// estimated (t_setup_ms: both); the target is resolved (with --target
// best-known the row of the table must have the n and m of the instance); --warmup runs with first
// replica 0 are not recorded; then repetition k = 0 .. reps - 1 runs the solver with first_replica
// = k*R and the same seed, so different global replicas and different initial conditions
// (decision B7), appends one row to --out and, with --replica-out, R rows to the replica file, and
// prints one line "rep <k>: ..." on stdout. The solver is run_sb_sequential (dsb or bsb, c0),
// run_sb_openmp with T threads (dsb or bsb, c1; T = 0, the default, is the default number of
// OpenMP threads), run_sa_sequential (sa, c0) or run_sa_openmp with T threads (sa, c2).
// t_solve_ms is the duration of the call of the solver measured by sbmc_run; t_prep_ms, t_run_ms
// and t_check_ms are the three phases measured inside it (sbmc/solve_times.hpp: preparation, run,
// final check), whose sum is at most t_solve_ms; t_step_us is t_run_ms per step or sweep. The
// column threads is the team of the solver (1 for c0); dt_rule (auto or value) and c0_rule (sigma,
// lambda or value) say how dt and c0 of SB were chosen; lambda_min and lambda_max are the extremes
// of the estimate, empty without it; run_tag is the text of --tag.
//
// Exit status: 0 on success; 1 for errors of the library (preconditions such as --replicas 0),
// of reading or writing (an --out with another header than that of results schema 2 included),
// for an instance that is not in the best-known table or that has another n or m than its row,
// when the paths of --out and --replica-out cannot be compared, when OMP_PLACES or OMP_PROC_BIND
// contains a line break, and when dt or c0 cannot be derived from the spectrum (J = 0: before any
// run, no row written), and on Windows when the processors or the mask cannot be read or set; 2
// for errors of the command line, with a message on stderr.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <CLI11/CLI11.hpp>

#include <sbmc/build_info.hpp>
#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/results.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sa.hpp>
#include <sbmc/sa_openmp.hpp>
#include <sbmc/sa_sequential.hpp>
#include <sbmc/sb.hpp>
#include <sbmc/sb_openmp.hpp>
#include <sbmc/sb_sequential.hpp>
#include <sbmc/solve_times.hpp>
#include <sbmc/spectrum.hpp>

#ifdef _WIN32
#include "win_platform.hpp"
#endif

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// Values of the command line, as given.
struct Options {
  std::string instance_file;
  std::int64_t dense_n = 0;
  std::uint64_t instance_seed = 0;
  std::string name;
  std::string algo;
  int replicas = 0;
  int steps = 0;   // dsb and bsb
  std::string dt;  // auto or a number
  std::string dt_factor = "1";
  int sweeps = 0;     // sa
  std::string beta0;  // sa
  std::string beta1;  // sa
  std::string variant = "c0";
  int threads = 0;  // c1 and c2 only
  std::string a0 = "1";
  std::string c0 = "auto";
  int eval_every = 0;
  std::uint64_t seed = 0;
  int reps = 1;
  int warmup = 1;
  std::string target = "none";
  std::string best_known = "data/best_known.csv";
  std::string out;
  std::string replica_out;
  std::string notes;
  std::string tag;
  std::string affinity;         // Windows only
  bool print_topology = false;  // Windows only
};

// Float over the whole text, as std::from_chars reads it (shortest round trip, no locale).
std::optional<float> to_float(std::string_view text) {
  float value = 0.0f;
  const char* const last = text.data() + text.size();
  const auto [end, error] = std::from_chars(text.data(), last, value);
  if (text.empty() || error != std::errc{} || end != last) {
    return std::nullopt;
  }
  return value;
}

// Double over the whole text, as std::from_chars reads it (no locale).
std::optional<double> to_double(std::string_view text) {
  double value = 0.0;
  const char* const last = text.data() + text.size();
  const auto [end, error] = std::from_chars(text.data(), last, value);
  if (text.empty() || error != std::errc{} || end != last) {
    return std::nullopt;
  }
  return value;
}

// Decimal int64 over the whole text, with an optional '-'.
std::optional<std::int64_t> to_int64(std::string_view text) {
  std::int64_t value = 0;
  const char* const last = text.data() + text.size();
  const auto [end, error] = std::from_chars(text.data(), last, value);
  if (text.empty() || error != std::errc{} || end != last) {
    return std::nullopt;
  }
  return value;
}

bool has_line_break(std::string_view text) {
  return text.find_first_of("\n\r") != std::string_view::npos;
}

// CLI11 converts integers with strtoll and strtoull in base 0 ("010" is octal, "0x10"
// hexadecimal) and also accepts digit separators and "true": the integer options of sbmc_run take
// decimal integers only. This transform rejects anything else and removes leading zeros, so that
// CLI11 reads the value in base 10; CLI11 still checks the range of the destination type.
CLI::Validator decimal_integer() {
  return {[](std::string& value) -> std::string {
            const std::size_t sign = !value.empty() && (value[0] == '-' || value[0] == '+') ? 1 : 0;
            if (value.size() == sign ||
                value.find_first_not_of("0123456789", sign) != std::string::npos) {
              return "'" + value + "' is not a decimal integer";
            }
            const std::size_t first = value.find_first_not_of('0', sign);
            value =
                value.substr(0, sign) + (first == std::string::npos ? "0" : value.substr(first));
            return {};
          },
          "", "DECIMAL"};
}

CLI::Validator float_value() {
  return {[](std::string& value) -> std::string {
            return to_float(value) ? std::string{} : "'" + value + "' is not a number";
          },
          "", "FLOAT"};
}

// A double, finite and > 0 (the inverse temperatures of SA).
CLI::Validator finite_positive_double() {
  return {[](std::string& value) -> std::string {
            const std::optional<double> number = to_double(value);
            return number && std::isfinite(*number) && *number > 0.0
                       ? std::string{}
                       : "expected a finite number > 0, got '" + value + "'";
          },
          "", "X"};
}

// Formats a float in the shortest form that reads back as the same float.
std::string shortest(float value) {
  std::array<char, 64> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("nan");
}

std::string shortest(double value) {
  std::array<char, 64> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("nan");
}

// Fixed notation with three decimals.
std::string fixed3(double value) {
  std::array<char, 512> text{};
  const auto [end, error] =
      std::to_chars(text.data(), text.data() + text.size(), value, std::chars_format::fixed, 3);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("nan");
}

// UTC time in the form YYYY-MM-DDThh:mm:ssZ.
std::string utc_timestamp(std::chrono::system_clock::time_point when) {
  const std::time_t seconds = std::chrono::system_clock::to_time_t(when);
  const std::tm* const utc = std::gmtime(&seconds);
  std::array<char, 32> text{};
  if (utc == nullptr || std::strftime(text.data(), text.size(), "%Y-%m-%dT%H:%M:%SZ", utc) == 0) {
    return "unknown";
  }
  return text.data();
}

#ifndef _WIN32
std::string trim(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

// First line of a small system file, trimmed; "unknown" when it is missing or empty.
std::string first_line_of(const std::filesystem::path& file) {
  std::ifstream in(file);
  std::string line;
  if (!in || !std::getline(in, line)) {
    return "unknown";
  }
  line = trim(std::move(line));
  return line.empty() ? "unknown" : line;
}

// PRETTY_NAME of /etc/os-release, without quotes; "unknown" when missing.
std::string os_pretty_name() {
  std::ifstream in("/etc/os-release");
  std::string line;
  constexpr std::string_view kKey = "PRETTY_NAME=";
  while (in && std::getline(in, line)) {
    if (line.compare(0, kKey.size(), kKey) != 0) {
      continue;
    }
    std::string value = trim(line.substr(kKey.size()));
    if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
        value.back() == value.front()) {
      value = value.substr(1, value.size() - 2);
    }
    std::string unescaped;
    for (std::size_t k = 0; k < value.size(); ++k) {
      if (value[k] == '\\' && k + 1 < value.size()) {
        ++k;
      }
      unescaped.push_back(value[k]);
    }
    unescaped = trim(std::move(unescaped));
    return unescaped.empty() ? "unknown" : unescaped;
  }
  return "unknown";
}
#endif  // !_WIN32

// One row of results schema 2, filled by column name.
class ResultsRow {
 public:
  ResultsRow() : values_(sbmc::results_columns().size()) {}

  void set(std::string_view column, std::string value) {
    const std::span<const std::string_view> columns = sbmc::results_columns();
    for (std::size_t k = 0; k < columns.size(); ++k) {
      if (columns[k] == column) {
        values_[k] = std::move(value);
        return;
      }
    }
    throw sbmc::Error("sbmc_run: internal error: no column '" + std::string(column) +
                      "' in results schema 2");
  }

  [[nodiscard]] const std::vector<std::string>& values() const noexcept { return values_; }

 private:
  std::vector<std::string> values_;
};

// The checks of sbmc::append_results_row and sbmc::append_replica_rows done before any run, so
// that a wrong --out or --replica-out does not waste a run: an existing non-empty file must start
// with the header of its schema (`columns`, named `what` in the messages) and end with '\n'; a
// new file needs an existing folder.
void check_output_file(const std::filesystem::path& file, std::string_view option,
                       std::span<const std::string_view> columns, std::string_view what) {
  const std::string name = std::string(option) + " " + file.string();
  std::error_code error;
  if (!std::filesystem::exists(file, error)) {
    const std::filesystem::path folder = file.parent_path();
    if (!folder.empty() && !std::filesystem::is_directory(folder, error)) {
      throw sbmc::IoError(name + ": the folder " + folder.string() + " does not exist");
    }
    return;
  }
  if (std::filesystem::is_directory(file, error)) {
    throw sbmc::IoError(name + " is a directory");
  }
  std::ifstream in(file, std::ios::binary);
  std::string first;
  if (!in) {
    throw sbmc::IoError(name + ": cannot read the file");
  }
  if (!std::getline(in, first)) {
    return;  // empty file: the header is written first
  }
  const std::vector<std::string> names(columns.begin(), columns.end());
  if (first != sbmc::csv_line(names)) {
    throw sbmc::IoError(name + ": the first line is not the header of " + std::string(what) +
                        "; the file was not modified");
  }
  in.clear();
  in.seekg(-1, std::ios::end);
  char last = '\0';
  if (!in.get(last) || last != '\n') {
    throw sbmc::IoError(name + " does not end with a line break; the file was not modified");
  }
}

#ifndef _WIN32
// The text after "Cpus_allowed_list:" in its line of /proc/self/status, without leading and
// trailing spaces and tabs: the logical processors allowed to the process, in the cpulist format
// (for example 0-15 or 0-3,8). Empty when the file or the line is missing.
std::string cpu_affinity() {
  std::ifstream in("/proc/self/status");
  std::string line;
  constexpr std::string_view kKey = "Cpus_allowed_list:";
  while (in && std::getline(in, line)) {
    if (line.compare(0, kKey.size(), kKey) != 0) {
      continue;
    }
    const std::string value = line.substr(kKey.size());
    const std::size_t first = value.find_first_not_of(" \t");
    if (first == std::string::npos) {
      return {};
    }
    const std::size_t last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
  }
  return {};
}

// The columns of the process environment on Linux: cpu_affinity, cpu_classes (empty), host and
// os.
struct ProcessEnvironment {
  std::string cpu_affinity;
  std::string cpu_classes;
  std::string host;
  std::string os;
};

ProcessEnvironment process_environment() {
  return {cpu_affinity(), std::string(), first_line_of("/proc/sys/kernel/hostname"),
          os_pretty_name() + " / " + first_line_of("/proc/sys/kernel/osrelease")};
}
#else
using sbmc_win::process_environment;
#endif  // !_WIN32

// The value of an environment variable, empty when it does not exist; a line break in the value
// is an error (one CSV row per line).
std::string environment_variable(const char* variable) {
  const char* const value = std::getenv(variable);
  if (value == nullptr) {
    return {};
  }
  std::string text(value);
  if (has_line_break(text)) {
    throw sbmc::Error("the environment variable " + std::string(variable) +
                      " contains a line break, which a CSV row cannot hold");
  }
  return text;
}

struct Target {
  std::optional<std::int64_t> value;
  std::string source = "none";  // best_known | user | none
};

// What a call of the solver gives to the rows: per replica best_cut, best_step (best_sweep of SA)
// and final_cut; the team of the solver; the c0 actually used (SB only); t_solve_ms; the times
// measured inside the solver.
struct ReplicaOutcome {
  std::int64_t best_cut = 0;
  std::int64_t best_step = 0;
  std::int64_t final_cut = 0;
};

struct SolveOutcome {
  std::vector<ReplicaOutcome> replicas;
  int threads = 1;
  std::string c0;
  double t_solve_ms = 0.0;
  sbmc::SolveTimes times;
};

int run(const Options& options, bool dense) {
  const bool sa = options.algo == "sa";

  // 0. The environment of the process, read once at startup, before any run.
  const auto environment = process_environment();
  const std::string omp_places = environment_variable("OMP_PLACES");
  const std::string omp_proc_bind = environment_variable("OMP_PROC_BIND");

  // 1. Instance; t_setup_ms is the time of this phase.
  const auto setup_start = std::chrono::steady_clock::now();
  sbmc::Instance instance;
  std::int64_t nnz = 0;
  if (dense) {
    instance = sbmc::DenseSpinGlass{options.dense_n, options.instance_seed};
  } else {
    sbmc::Graph graph = sbmc::read_rudy(options.instance_file);
    nnz = static_cast<std::int64_t>(graph.edges.size());
    instance = std::move(graph);
  }
  const std::int64_t n = sbmc::num_spins(instance);  // checks n of a DenseSpinGlass
  if (dense) {
    nnz = n * (n - 1) / 2;
  }
  // The spectrum of J, once, only when dt or c0 is derived from it; with J = 0 spectral_dt and
  // spectral_c0 throw InvalidArgument here, before any run.
  std::optional<sbmc::SpectrumEstimate> spectrum;
  std::optional<float> spectral_dt;
  std::optional<float> spectral_c0;
  if (!sa && (options.dt == "auto" || options.c0 == "lambda")) {
    spectrum = sbmc::estimate_spectrum(instance, sbmc::kSpectrumIterations);
    if (options.dt == "auto") {
      spectral_dt = sbmc::spectral_dt(*spectrum, *to_double(options.dt_factor));
    }
    if (options.c0 == "lambda") {
      spectral_c0 = sbmc::spectral_c0(*spectrum);
    }
  }
  const double t_setup_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start)
          .count();

  std::string name = options.name;
  if (name.empty()) {
    name = dense ? "dense_n" + std::to_string(options.dense_n) + "_s" +
                       std::to_string(options.instance_seed)
                 : std::filesystem::path(options.instance_file).stem().string();
  }

  // 2. Target.
  Target target;
  if (options.target == "best-known") {
    const std::vector<sbmc::BestKnownEntry> table = sbmc::read_best_known(options.best_known);
    const sbmc::BestKnownEntry* const entry = sbmc::find_best_known(table, name);
    if (entry == nullptr) {
      throw sbmc::Error("--target best-known: instance '" + name + "' is not in " +
                        options.best_known + " (choose the row with --name)");
    }
    if (entry->n != n || entry->m != nnz) {
      throw sbmc::Error("--target best-known: " + name + " in " + options.best_known + " has n " +
                        std::to_string(entry->n) + " and m " + std::to_string(entry->m) +
                        ", the instance has n " + std::to_string(n) + " and " +
                        std::to_string(nnz) + " edges");
    }
    target = {entry->best_known, "best_known"};
  } else if (options.target != "none") {
    target = {to_int64(options.target), "user"};
  }

  // The checks that need no run.
  if (!options.out.empty()) {
    check_output_file(options.out, "--out", sbmc::results_columns(), "results schema 2");
  }
  if (!options.replica_out.empty()) {
    check_output_file(options.replica_out, "--replica-out", sbmc::replica_columns(),
                      "the replica file (schema 1)");
  }
  if (static_cast<std::int64_t>(options.reps) * options.replicas > (std::int64_t{1} << 32)) {
    throw sbmc::InvalidArgument("--reps " + std::to_string(options.reps) + " with --replicas " +
                                std::to_string(options.replicas) +
                                " needs global replicas beyond 2^32");
  }

  // Parameters of the solver: SB (dsb, bsb) or SA.
  sbmc::SbParams sb_params;
  sbmc::SaParams sa_params;
  if (sa) {
    sa_params.replicas = options.replicas;
    sa_params.sweeps = options.sweeps;
    sa_params.beta0 = *to_double(options.beta0);
    sa_params.beta1 = *to_double(options.beta1);
    sa_params.seed = options.seed;
  } else {
    sb_params.algorithm =
        options.algo == "dsb" ? sbmc::SbAlgorithm::discrete : sbmc::SbAlgorithm::ballistic;
    sb_params.replicas = options.replicas;
    sb_params.steps = options.steps;
    sb_params.dt = spectral_dt ? *spectral_dt : *to_float(options.dt);
    sb_params.a0 = *to_float(options.a0);
    if (options.c0 == "auto") {
      sb_params.c0 = 0.0f;  // automatic in the library: auto_c0, the rule sigma
    } else {
      sb_params.c0 = spectral_c0 ? *spectral_c0 : *to_float(options.c0);
    }
    sb_params.eval_every = options.eval_every;
    sb_params.seed = options.seed;
  }
  const int steps = sa ? options.sweeps : options.steps;  // the column steps: M or S

  // The solver of the algorithm and of the variant; t_solve_ms is the duration of its call.
  const auto solve = [&](std::int64_t first_replica) {
    SolveOutcome outcome;
    if (sa) {
      sa_params.first_replica = first_replica;
      const auto start = std::chrono::steady_clock::now();
      const sbmc::SaResult result = options.variant == "c2"
                                        ? sbmc::run_sa_openmp(instance, sa_params, options.threads)
                                        : sbmc::run_sa_sequential(instance, sa_params);
      outcome.t_solve_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
      outcome.threads = result.threads;  // 1 for C0
      outcome.times = result.times;
      for (const sbmc::SaReplicaResult& r : result.per_replica) {
        outcome.replicas.push_back({r.best_cut, r.best_sweep, r.final_cut});
      }
    } else {
      sb_params.first_replica = first_replica;
      const auto start = std::chrono::steady_clock::now();
      const sbmc::SbResult result = options.variant == "c1"
                                        ? sbmc::run_sb_openmp(instance, sb_params, options.threads)
                                        : sbmc::run_sb_sequential(instance, sb_params);
      outcome.t_solve_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
      outcome.threads = result.threads;  // 1 for C0
      outcome.times = result.times;
      outcome.c0 = shortest(result.c0);
      for (const sbmc::SbReplicaResult& r : result.per_replica) {
        outcome.replicas.push_back({r.best_cut, r.best_step, r.final_cut});
      }
    }
    return outcome;
  };

  // 3. Warm-up runs, not recorded.
  for (int w = 0; w < options.warmup; ++w) {
    static_cast<void>(solve(0));
  }

  // Columns that do not change between repetitions.
  const std::string dirty =
      sbmc::build_info::dirty < 0 ? std::string() : std::to_string(sbmc::build_info::dirty);

  // 4. Recorded repetitions.
  for (int k = 0; k < options.reps; ++k) {
    const std::int64_t first_replica = static_cast<std::int64_t>(k) * options.replicas;
    const std::string timestamp = utc_timestamp(std::chrono::system_clock::now());
    const SolveOutcome outcome = solve(first_replica);
    const double t_solve_ms = outcome.t_solve_ms;
    const std::string t_run_ms = fixed3(outcome.times.run_ms);
    const std::string t_check_ms = fixed3(outcome.times.check_ms);

    std::int64_t best_cut = std::numeric_limits<std::int64_t>::min();
    std::int64_t sum = 0;
    std::int64_t successes = 0;
    for (const ReplicaOutcome& replica : outcome.replicas) {
      best_cut = std::max(best_cut, replica.best_cut);
      if ((replica.best_cut > 0 &&
           sum > std::numeric_limits<std::int64_t>::max() - replica.best_cut) ||
          (replica.best_cut < 0 &&
           sum < std::numeric_limits<std::int64_t>::min() - replica.best_cut)) {
        throw sbmc::Error("sbmc_run: the sum of the cuts overflows int64");
      }
      sum += replica.best_cut;
      if (target.value && replica.best_cut >= *target.value) {
        ++successes;
      }
    }
    const std::string mean_cut =
        fixed3(static_cast<double>(sum) / static_cast<double>(options.replicas));

    ResultsRow row;
    row.set("schema_version", std::to_string(sbmc::kResultsSchemaVersion));
    row.set("timestamp", timestamp);
    row.set("git_hash", std::string(sbmc::build_info::git_hash));
    row.set("dirty", dirty);
    row.set("host", environment.host);
    row.set("os", environment.os);
    row.set("compiler", std::string(sbmc::build_info::compiler));
    row.set("algo", options.algo);
    row.set("variant", options.variant);
    row.set("instance", name);
    row.set("n", std::to_string(n));
    row.set("nnz", std::to_string(nnz));
    row.set("replicas", std::to_string(options.replicas));
    row.set("steps", std::to_string(steps));
    if (sa) {
      row.set("beta0", shortest(sa_params.beta0));
      row.set("beta1", shortest(sa_params.beta1));
    } else {
      row.set("dt", shortest(sb_params.dt));
      row.set("c0", outcome.c0);
    }
    row.set("seed", std::to_string(options.seed));
    row.set("rep", std::to_string(k));
    row.set("t_setup_ms", fixed3(t_setup_ms));
    row.set("t_solve_ms", fixed3(t_solve_ms));
    row.set("t_step_us", fixed3(outcome.times.run_ms * 1000.0 / static_cast<double>(steps)));
    row.set("best_cut", std::to_string(best_cut));
    row.set("mean_cut", mean_cut);
    if (target.value) {
      row.set("n_success", std::to_string(successes));
      row.set("p_s",
              shortest(static_cast<double>(successes) / static_cast<double>(options.replicas)));
      row.set("target", std::to_string(*target.value));
    }
    row.set("ranks", "1");
    if (!sa) {
      row.set("a0", shortest(sb_params.a0));
      row.set("eval_every", std::to_string(options.eval_every));
    }
    row.set("first_replica", std::to_string(first_replica));
    row.set("threads", std::to_string(outcome.threads));  // 1 for c0
    row.set("target_source", target.source);
    row.set("t_prep_ms", fixed3(outcome.times.prep_ms));
    row.set("t_run_ms", t_run_ms);
    row.set("t_check_ms", t_check_ms);
    if (!sa) {
      row.set("dt_rule", options.dt == "auto" ? "auto" : "value");
      row.set("c0_rule", options.c0 == "auto"     ? "sigma"
                         : options.c0 == "lambda" ? "lambda"
                                                  : "value");
    }
    if (spectrum) {
      row.set("lambda_min", shortest(spectrum->lambda_min));
      row.set("lambda_max", shortest(spectrum->lambda_max));
    }
    row.set("cpu_affinity", environment.cpu_affinity);
    row.set("cpu_classes", environment.cpu_classes);
    row.set("omp_places", omp_places);
    row.set("omp_proc_bind", omp_proc_bind);
    row.set("run_tag", options.tag);
    row.set("notes", options.notes);
    if (!options.out.empty()) {
      sbmc::append_results_row(options.out, row.values());
    }
    if (!options.replica_out.empty()) {
      std::vector<std::vector<std::string>> replica_rows;
      replica_rows.reserve(outcome.replicas.size());
      for (std::size_t r = 0; r < outcome.replicas.size(); ++r) {
        const ReplicaOutcome& replica = outcome.replicas[r];
        replica_rows.push_back({std::to_string(sbmc::kReplicaSchemaVersion), timestamp, name,
                                options.algo, options.variant, std::to_string(options.replicas),
                                std::to_string(steps), std::to_string(options.seed),
                                std::to_string(k),
                                std::to_string(first_replica + static_cast<std::int64_t>(r)),
                                std::to_string(replica.best_cut), std::to_string(replica.best_step),
                                std::to_string(replica.final_cut)});
      }
      sbmc::append_replica_rows(options.replica_out, replica_rows);
    }

    std::cout << "rep " << k << ": best_cut " << best_cut << " mean_cut " << mean_cut;
    if (target.value) {
      std::cout << " n_success " << successes << "/" << options.replicas;
    }
    std::cout << " t_solve_ms " << fixed3(t_solve_ms) << " t_run_ms " << t_run_ms << " t_check_ms "
              << t_check_ms << '\n'
              << std::flush;
  }
  return kExitOk;
}

int usage_error(std::string_view message) {
  std::cerr << "sbmc_run: " << message << "\nRun with --help for more information.\n";
  return kExitUsage;
}

std::string version_text() {
  return "sbmc_run git_hash=" + std::string(sbmc::build_info::git_hash) +
         " dirty=" + std::to_string(sbmc::build_info::dirty) +
         " compiler=" + std::string(sbmc::build_info::compiler) +
         " build_type=" + std::string(sbmc::build_info::build_type);
}

// --affinity: the process mask, before any call of OpenMP (Windows only).
int apply_affinity(const std::string& affinity) {
#ifdef _WIN32
  try {
    sbmc_win::apply_affinity(affinity);
    return kExitOk;
  } catch (const std::invalid_argument& e) {
    return usage_error(e.what());
  } catch (const std::exception& e) {
    std::cerr << "sbmc_run: --affinity " << affinity << ": " << e.what() << '\n';
    return kExitError;
  }
#else
  static_cast<void>(affinity);
  return usage_error("--affinity is available only on Windows");
#endif
}

// --print-topology, after --affinity when given (Windows only).
int print_topology(const std::string& affinity) {
#ifdef _WIN32
  if (!affinity.empty()) {
    if (const int status = apply_affinity(affinity); status != kExitOk) {
      return status;
    }
  }
  try {
    sbmc_win::print_topology(std::cout);
    return kExitOk;
  } catch (const std::exception& e) {
    std::cerr << "sbmc_run: --print-topology: " << e.what() << '\n';
    return kExitError;
  }
#else
  static_cast<void>(affinity);
  return usage_error("--print-topology is available only on Windows");
#endif
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
  sbmc_win::init_stdout();
#endif
  Options options;
  CLI::App app{
      "Runs simulated bifurcation (dsb or bsb: the sequential reference C0, or C1 with OpenMP over "
      "blocks of replicas) or simulated annealing (sa: the sequential reference C0, or C2 with "
      "OpenMP over blocks of replicas) on one instance, and writes one CSV row per repetition "
      "(results schema 2) and, with --replica-out, one row per replica (replica file, schema 1).",
      "sbmc_run"};
  app.failure_message([](const CLI::App* parser, const CLI::Error& e) {
    const std::string message = CLI::FailureMessage::simple(parser, e);
    return message.starts_with("sbmc_run: ") ? message : "sbmc_run: " + message;
  });
  app.footer(
      "dsb and bsb need --steps and --dt and take the variants c0 and c1; sa needs --sweeps, "
      "--beta0 and --beta1, takes the variants c0 and c2, and takes neither --steps, --dt, --c0, "
      "--a0 nor --eval-every. Repetition k runs the solver with first_replica = k * R and the same "
      "--seed: different global replicas, hence different initial conditions. --threads only with "
      "--variant c1 or c2; the column threads is the team of the solver (1 for c0). --replica-out "
      "(with --out, on another file) appends one row per replica of every repetition. --target "
      "best-known needs a row of the table with the n and m of the instance. Times: t_solve_ms is "
      "the duration of the call of the solver; t_prep_ms, t_run_ms and t_check_ms are its "
      "preparation, run and final check, measured inside it; t_step_us is t_run_ms per step or "
      "sweep. --tag fills the column run_tag; cpu_affinity, omp_places and omp_proc_bind record "
      "the allowed processors and OMP_PLACES and OMP_PROC_BIND at startup. --dt auto (dt = F * "
      "1.25 sqrt(2 / (1 - lambda_min / lambda_max)), F = --dt-factor) and --c0 lambda (c0 = 1 / "
      "lambda_max) estimate the spectrum of J once, with 100 Lanczos steps, after reading the "
      "instance (inside t_setup_ms), and fill lambda_min and lambda_max. Exit status: 0 success; "
      "1 error of the library, of reading or writing (an --out with another header than that of "
      "results schema 2 included), instance not in the best-known table or with another n or m, "
      "paths of --out and --replica-out that cannot be compared, OMP_PLACES or OMP_PROC_BIND with "
      "a line break, dt or c0 not defined by the spectrum (J = 0); 2 error of the command "
      "line.");

  const CLI::Validator decimal = decimal_integer();
  const CLI::Validator number = float_value();
  const CLI::Validator positive = finite_positive_double();
  const auto no_line_break = [](const std::string& value) -> std::string {
    return has_line_break(value) ? "line breaks are not allowed (one CSV row per line)"
                                 : std::string{};
  };

  CLI::Option* instance =
      app.add_option("--instance", options.instance_file, "Instance: rudy file (G-set format)")
          ->type_name("FILE");
  CLI::Option* dense =
      app.add_option("--dense", options.dense_n,
                     "Instance: DenseSpinGlass with N spins, J_ij = +-1 from --instance-seed")
          ->type_name("N")
          ->transform(decimal);
  CLI::Option* instance_seed =
      app.add_option("--instance-seed", options.instance_seed, "Seed of the couplings of --dense")
          ->type_name("S")
          ->transform(decimal);
  instance->excludes(dense);
  dense->needs(instance_seed);
  instance_seed->needs(dense);
  app.add_option("--name", options.name,
                 "Instance name in the CSV and in the best-known table (default: the file name "
                 "without folder and extension, or dense_n<N>_s<S>)")
      ->check([](const std::string& value) -> std::string {
        if (value.empty()) {
          return "the name cannot be empty";
        }
        return has_line_break(value) ? "line breaks are not allowed" : std::string{};
      });
  app.add_option("--algo", options.algo,
                 "Algorithm: dsb or bsb (simulated bifurcation), sa (simulated annealing)")
      ->required()
      ->check(CLI::IsMember({"dsb", "bsb", "sa"}));
  app.add_option("--replicas", options.replicas, "Replicas R per repetition")
      ->required()
      ->transform(decimal);
  CLI::Option* steps =
      app.add_option("--steps", options.steps, "dsb and bsb: steps M")->transform(decimal);
  CLI::Option* dt =
      app.add_option("--dt", options.dt,
                     "dsb and bsb: time step, auto (--dt-factor * 1.25 sqrt(2 / (1 - lambda_min / "
                     "lambda_max)), from the estimate of the spectrum of J) or a finite number > 0 "
                     "(no default: 1.25 suits random +-1 matrices and fails on graphs with "
                     "positive weights)")
          ->type_name("auto|X")
          ->check([](const std::string& value) -> std::string {
            if (value == "auto") {
              return {};
            }
            const std::optional<float> number = to_float(value);
            return number && std::isfinite(*number) && *number > 0.0f
                       ? std::string{}
                       : "expected auto or a finite number > 0, got '" + value + "'";
          });
  CLI::Option* dt_factor =
      app.add_option(
             "--dt-factor", options.dt_factor,
             "dsb and bsb with --dt auto: factor F of the automatic dt, a finite number > 0")
          ->capture_default_str()
          ->type_name("F")
          ->check(positive);
  CLI::Option* sweeps = app.add_option("--sweeps", options.sweeps, "sa: sweeps S")
                            ->type_name("S")
                            ->transform(decimal)
                            ->check(CLI::Range(1, std::numeric_limits<int>::max()));
  CLI::Option* beta0 =
      app.add_option("--beta0", options.beta0,
                     "sa: inverse temperature of the first sweep, a finite number > 0")
          ->check(positive);
  CLI::Option* beta1 =
      app.add_option("--beta1", options.beta1,
                     "sa: inverse temperature of the last sweep, finite and >= --beta0 "
                     "(geometric schedule in between)")
          ->check(positive);
  app.add_option("--variant", options.variant,
                 "Variant: c0: the sequential reference; c1: OpenMP over blocks of replicas (dsb "
                 "and bsb); c2: OpenMP over blocks of replicas (sa)")
      ->capture_default_str()
      ->check(CLI::IsMember({"c0", "c1", "c2"}));
  CLI::Option* threads =
      app.add_option("--threads", options.threads,
                     "OpenMP threads of c1 and c2 (0: the default number of OpenMP threads; at "
                     "most 1024)")
          ->capture_default_str()
          ->type_name("T")
          ->transform(decimal)
          ->check(CLI::Range(0, 1024));
  CLI::Option* a0 = app.add_option("--a0", options.a0, "dsb and bsb: a0")
                        ->capture_default_str()
                        ->type_name("X")
                        ->check(number);
  CLI::Option* c0 =
      app.add_option("--c0", options.c0,
                     "dsb and bsb: c0, auto (1/(2 sigma sqrt(n))), lambda (1/lambda_max, from the "
                     "estimate of the spectrum of J) or a finite number > 0")
          ->capture_default_str()
          ->type_name("auto|lambda|X")
          ->check([](const std::string& value) -> std::string {
            if (value == "auto" || value == "lambda") {
              return {};
            }
            const std::optional<float> c0 = to_float(value);
            if (!c0 || !std::isfinite(*c0) || !(*c0 > 0.0f)) {
              return "expected auto, lambda or a finite number > 0, got '" + value +
                     "' (0 is not accepted: in the library c0 = 0 means automatic)";
            }
            return {};
          });
  CLI::Option* eval_every =
      app.add_option("--eval-every", options.eval_every,
                     "bsb: evaluate the signs every K steps (0: only the final configuration)")
          ->capture_default_str()
          ->type_name("K")
          ->transform(decimal);
  app.add_option("--seed", options.seed,
                 "Seed of the initial positions (dsb, bsb) or of the initial spins and of the "
                 "acceptance words (sa)")
      ->capture_default_str()
      ->type_name("S")
      ->transform(decimal);
  app.add_option("--reps", options.reps, "Recorded repetitions")
      ->capture_default_str()
      ->type_name("K")
      ->transform(decimal)
      ->check(CLI::Range(1, std::numeric_limits<int>::max()));
  app.add_option("--warmup", options.warmup, "Warm-up runs, not recorded")
      ->capture_default_str()
      ->type_name("K")
      ->transform(decimal)
      ->check(CLI::Range(0, std::numeric_limits<int>::max()));
  app.add_option("--target", options.target,
                 "Target cut: best-known (row --name of --best-known), none, or an integer")
      ->capture_default_str()
      ->type_name("best-known|none|INTEGER")
      ->check([](const std::string& value) -> std::string {
        return value == "best-known" || value == "none" || to_int64(value)
                   ? std::string{}
                   : "expected best-known, none or a decimal integer, got '" + value + "'";
      });
  app.add_option("--best-known", options.best_known,
                 "Table of the best-known values, relative to the current folder")
      ->capture_default_str()
      ->type_name("FILE");
  app.add_option("--out", options.out,
                 "CSV file of the results: one row appended per repetition (default: no CSV)")
      ->type_name("FILE");
  CLI::Option* replica_out =
      app.add_option("--replica-out", options.replica_out,
                     "CSV file of the replicas: R rows appended per repetition (replica file, "
                     "schema 1); requires --out")
          ->type_name("FILE");
  app.add_option("--notes", options.notes, "Free text of the notes column")
      ->type_name("TEXT")
      ->check(no_line_break);
  app.add_option("--tag", options.tag,
                 "Free text of the run_tag column: the identifier of the campaign or of the "
                 "configuration")
      ->type_name("TEXT")
      ->check(no_line_break);
  // Phase 07, part 4. --print-topology stops the parsing (CLI::Success) in its callback, run
  // with the priority First of CLI11, before the required options are checked; --affinity has
  // the same priority and is defined first, so its value is stored by then.
  CLI::Option* affinity =
      app.add_option("--affinity", options.affinity,
                     "Windows only: process mask set before any OpenMP call: p1 (one logical "
                     "processor per P core), p (P cores), e (E cores) or all")
          ->type_name("p1|p|e|all")
          ->check(CLI::IsMember({"p1", "p", "e", "all"}))
          ->callback_priority(CLI::CallbackPriority::First);
  app.add_flag_callback(
         "--print-topology",
         [&options]() {
           options.print_topology = true;
           throw CLI::Success();
         },
         "Windows only: write the logical processors, their classes and masks, the effective mask "
         "and the OpenMP view, after --affinity when given, and exit")
      ->callback_priority(CLI::CallbackPriority::First);
  app.set_version_flag("--version", version_text(), "Write the build information and exit");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    if (options.print_topology) {
      return print_topology(options.affinity);
    }
    return app.exit(e) == kExitOk ? kExitOk : kExitUsage;
  }
  // The process mask first, before any other work and any call of OpenMP.
  if (affinity->count() > 0) {
    if (const int status = apply_affinity(options.affinity); status != kExitOk) {
      return status;
    }
  }
  // The rules of the algorithms, after the parsing and before the other checks of the command line
  // (until phase 05 --steps and --dt were required options of CLI11, checked at this point of the
  // order, with the same messages).
  if (options.algo == "sa") {
    for (const auto& [option, name] : {std::pair{sweeps, "--sweeps"}, std::pair{beta0, "--beta0"},
                                       std::pair{beta1, "--beta1"}}) {
      if (option->count() == 0) {
        return usage_error(std::string(name) + " is required with --algo sa");
      }
    }
    for (const auto& [option, name] :
         {std::pair{steps, "--steps"}, std::pair{dt, "--dt"}, std::pair{dt_factor, "--dt-factor"},
          std::pair{c0, "--c0"}, std::pair{a0, "--a0"}, std::pair{eval_every, "--eval-every"}}) {
      if (option->count() > 0) {
        return usage_error(std::string(name) + " is not used with --algo sa");
      }
    }
    if (options.variant == "c1") {
      return usage_error("--variant c1 is not available with --algo sa");
    }
    if (*to_double(options.beta1) < *to_double(options.beta0)) {
      return usage_error("--beta1 must be >= --beta0");
    }
  } else {
    for (const auto& [option, name] : {std::pair{steps, "--steps"}, std::pair{dt, "--dt"}}) {
      if (option->count() == 0) {
        return usage_error(std::string(name) + " is required");
      }
    }
    for (const auto& [option, name] : {std::pair{sweeps, "--sweeps"}, std::pair{beta0, "--beta0"},
                                       std::pair{beta1, "--beta1"}}) {
      if (option->count() > 0) {
        return usage_error(std::string(name) + " requires --algo sa");
      }
    }
    if (options.variant == "c2") {
      return usage_error("--variant c2 requires --algo sa");
    }
    if (dt_factor->count() > 0 && options.dt != "auto") {
      return usage_error("--dt-factor requires --dt auto");
    }
  }
  const bool use_dense = dense->count() > 0;
  if (instance->count() == 0 && !use_dense) {
    return usage_error("an instance is required: --instance FILE, or --dense N --instance-seed S");
  }
  if (threads->count() > 0 && options.variant == "c0") {
    return usage_error("--threads requires --variant c1 or c2");
  }
  if (replica_out->count() > 0 && options.out.empty()) {
    return usage_error("--replica-out requires --out");
  }
  if (replica_out->count() > 0) {
    // The same file written in two ways ("x.csv", "./x.csv", an absolute path) is the same file.
    // The paths are made absolute first: weakly_canonical leaves a relative path whose first
    // element does not exist (a new file in the current folder) relative.
    std::error_code error;
    const auto canonical = [&error](const std::string& file) {
      std::filesystem::path path = std::filesystem::absolute(file, error);
      return error ? path : std::filesystem::weakly_canonical(path, error);
    };
    const std::filesystem::path out = canonical(options.out);
    const std::filesystem::path replicas = error ? out : canonical(options.replica_out);
    if (error) {
      std::cerr << "sbmc_run: --out " << options.out << " and --replica-out " << options.replica_out
                << ": cannot compare the two paths: " << error.message() << '\n';
      return kExitError;
    }
    if (out == replicas) {
      return usage_error("--replica-out must differ from --out");
    }
  }

  try {
    return run(options, use_dense);
  } catch (const std::exception& e) {
    std::cerr << "sbmc_run: " << e.what() << '\n';
    return kExitError;
  }
}
