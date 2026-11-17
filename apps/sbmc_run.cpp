// sbmc_run: runs a solver of the project on one instance and writes one CSV row per repetition,
// with the results schema 1 (sbmc/results.hpp). Phase 04: the sequential reference C0 of
// simulated bifurcation (variant c0), dSB or bSB.
//
//   sbmc_run (--instance FILE | --dense N --instance-seed S) --algo dsb|bsb --replicas R
//            --steps M --dt X [--name NAME] [--variant c0] [--a0 X] [--c0 auto|X]
//            [--eval-every K] [--seed S] [--reps K] [--warmup K]
//            [--target best-known|none|INTEGER] [--best-known FILE] [--out FILE] [--notes TEXT]
//
// Run: the instance is read (t_setup_ms); the target is resolved; --warmup runs with first
// replica 0 are not recorded; then repetition k = 0 .. reps - 1 runs C0 with first_replica = k*R
// and the same seed, so different global replicas and different initial conditions (decision B7),
// appends one row to --out and prints one line "rep <k>: ..." on stdout. t_solve_ms is the
// duration of run_sb_sequential, its internal preparation included.
//
// Exit status: 0 on success; 1 for errors of the library (preconditions such as --replicas 0),
// of reading or writing, and for an instance that is not in the best-known table; 2 for errors
// of the command line, with a message on stderr.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
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
#include <sbmc/sb.hpp>
#include <sbmc/sb_sequential.hpp>

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
  int steps = 0;
  std::string dt;
  std::string variant = "c0";
  std::string a0 = "1";
  std::string c0 = "auto";
  int eval_every = 0;
  std::uint64_t seed = 0;
  int reps = 1;
  int warmup = 1;
  std::string target = "none";
  std::string best_known = "data/best_known.csv";
  std::string out;
  std::string notes;
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

// One row of results schema 1, filled by column name.
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
                      "' in results schema 1");
  }

  [[nodiscard]] const std::vector<std::string>& values() const noexcept { return values_; }

 private:
  std::vector<std::string> values_;
};

// The checks of sbmc::append_results_row done before any run, so that a wrong --out does not waste
// a run: an existing non-empty file must start with the header of schema 1 and end with '\n'; a new
// file needs an existing folder.
void check_results_file(const std::filesystem::path& file) {
  std::error_code error;
  if (!std::filesystem::exists(file, error)) {
    const std::filesystem::path folder = file.parent_path();
    if (!folder.empty() && !std::filesystem::is_directory(folder, error)) {
      throw sbmc::IoError("--out " + file.string() + ": the folder " + folder.string() +
                          " does not exist");
    }
    return;
  }
  if (std::filesystem::is_directory(file, error)) {
    throw sbmc::IoError("--out " + file.string() + " is a directory");
  }
  std::ifstream in(file, std::ios::binary);
  std::string first;
  if (!in) {
    throw sbmc::IoError("--out " + file.string() + ": cannot read the file");
  }
  if (!std::getline(in, first)) {
    return;  // empty file: append_results_row writes the header first
  }
  const std::span<const std::string_view> columns = sbmc::results_columns();
  const std::vector<std::string> names(columns.begin(), columns.end());
  if (first != sbmc::csv_line(names)) {
    throw sbmc::IoError("--out " + file.string() +
                        ": the first line is not the header of results schema 1; the file was "
                        "not modified");
  }
  in.clear();
  in.seekg(-1, std::ios::end);
  char last = '\0';
  if (!in.get(last) || last != '\n') {
    throw sbmc::IoError("--out " + file.string() +
                        " does not end with a line break; the file was not modified");
  }
}

struct Target {
  std::optional<std::int64_t> value;
  std::string source = "none";  // best_known | user | none
};

int run(const Options& options, bool dense) {
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
      std::cerr << "sbmc_run: warning: " << name << " in " << options.best_known << " has n "
                << entry->n << " and m " << entry->m << ", the instance has n " << n << " and "
                << nnz << " edges\n";
    }
    target = {entry->best_known, "best_known"};
  } else if (options.target != "none") {
    target = {to_int64(options.target), "user"};
  }

  // The checks that need no run.
  if (!options.out.empty()) {
    check_results_file(options.out);
  }
  if (static_cast<std::int64_t>(options.reps) * options.replicas > (std::int64_t{1} << 32)) {
    throw sbmc::InvalidArgument("--reps " + std::to_string(options.reps) + " with --replicas " +
                                std::to_string(options.replicas) +
                                " needs global replicas beyond 2^32");
  }

  sbmc::SbParams params;
  params.algorithm =
      options.algo == "dsb" ? sbmc::SbAlgorithm::discrete : sbmc::SbAlgorithm::ballistic;
  params.replicas = options.replicas;
  params.steps = options.steps;
  params.dt = *to_float(options.dt);
  params.a0 = *to_float(options.a0);
  params.c0 = options.c0 == "auto" ? 0.0f : *to_float(options.c0);
  params.eval_every = options.eval_every;
  params.seed = options.seed;

  // 3. Warm-up runs, not recorded.
  for (int w = 0; w < options.warmup; ++w) {
    params.first_replica = 0;
    static_cast<void>(sbmc::run_sb_sequential(instance, params));
  }

  // Columns that do not change between repetitions.
  const std::string host = first_line_of("/proc/sys/kernel/hostname");
  const std::string os = os_pretty_name() + " / " + first_line_of("/proc/sys/kernel/osrelease");
  const std::string dirty =
      sbmc::build_info::dirty < 0 ? std::string() : std::to_string(sbmc::build_info::dirty);

  // 4. Recorded repetitions.
  for (int k = 0; k < options.reps; ++k) {
    params.first_replica = static_cast<std::int64_t>(k) * options.replicas;
    const std::string timestamp = utc_timestamp(std::chrono::system_clock::now());
    const auto solve_start = std::chrono::steady_clock::now();
    const sbmc::SbResult result = sbmc::run_sb_sequential(instance, params);
    const double t_solve_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solve_start)
            .count();

    std::int64_t best_cut = std::numeric_limits<std::int64_t>::min();
    std::int64_t sum = 0;
    std::int64_t successes = 0;
    for (const sbmc::SbReplicaResult& replica : result.per_replica) {
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
    row.set("host", host);
    row.set("os", os);
    row.set("compiler", std::string(sbmc::build_info::compiler));
    row.set("algo", options.algo);
    row.set("variant", options.variant);
    row.set("instance", name);
    row.set("n", std::to_string(n));
    row.set("nnz", std::to_string(nnz));
    row.set("replicas", std::to_string(options.replicas));
    row.set("steps", std::to_string(options.steps));
    row.set("dt", shortest(params.dt));
    row.set("c0", shortest(result.c0));
    row.set("seed", std::to_string(options.seed));
    row.set("rep", std::to_string(k));
    row.set("t_setup_ms", fixed3(t_setup_ms));
    row.set("t_solve_ms", fixed3(t_solve_ms));
    row.set("t_step_us", fixed3(t_solve_ms * 1000.0 / static_cast<double>(options.steps)));
    row.set("best_cut", std::to_string(best_cut));
    row.set("mean_cut", mean_cut);
    if (target.value) {
      row.set("n_success", std::to_string(successes));
      row.set("p_s",
              shortest(static_cast<double>(successes) / static_cast<double>(options.replicas)));
      row.set("target", std::to_string(*target.value));
    }
    row.set("ranks", "1");
    row.set("a0", shortest(params.a0));
    row.set("eval_every", std::to_string(options.eval_every));
    row.set("first_replica", std::to_string(params.first_replica));
    row.set("threads", "1");  // C0 is sequential
    row.set("target_source", target.source);
    row.set("notes", options.notes);
    if (!options.out.empty()) {
      sbmc::append_results_row(options.out, row.values());
    }

    std::cout << "rep " << k << ": best_cut " << best_cut << " mean_cut " << mean_cut;
    if (target.value) {
      std::cout << " n_success " << successes << "/" << options.replicas;
    }
    std::cout << " t_solve_ms " << fixed3(t_solve_ms) << '\n' << std::flush;
  }
  return kExitOk;
}

int usage_error(std::string_view message) {
  std::cerr << "sbmc_run: " << message << "\nRun with --help for more information.\n";
  return kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  CLI::App app{
      "Runs the sequential reference C0 of simulated bifurcation (dSB or bSB) on one instance and "
      "writes one CSV row per repetition (results schema 1).",
      "sbmc_run"};
  app.failure_message([](const CLI::App* parser, const CLI::Error& e) {
    const std::string message = CLI::FailureMessage::simple(parser, e);
    return message.starts_with("sbmc_run: ") ? message : "sbmc_run: " + message;
  });
  app.footer(
      "Repetition k runs C0 with first_replica = k * R and the same --seed: different global "
      "replicas, hence different initial conditions. Exit status: 0 success; 1 error of the "
      "library, of reading or writing, or instance not in the best-known table; 2 error of the "
      "command line.");

  const CLI::Validator decimal = decimal_integer();
  const CLI::Validator number = float_value();
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
  app.add_option("--algo", options.algo, "Algorithm")
      ->required()
      ->check(CLI::IsMember({"dsb", "bsb"}));
  app.add_option("--replicas", options.replicas, "Replicas R per repetition")
      ->required()
      ->transform(decimal);
  app.add_option("--steps", options.steps, "Steps M")->required()->transform(decimal);
  app.add_option("--dt", options.dt,
                 "Time step (no default: 1.25 suits random +-1 matrices and fails on graphs "
                 "with positive weights; automatic choice in phase 07)")
      ->required()
      ->type_name("X")
      ->check(number);
  app.add_option("--variant", options.variant, "Variant (phase 04: c0, the sequential reference)")
      ->capture_default_str()
      ->check(CLI::IsMember({"c0"}));
  app.add_option("--a0", options.a0, "a0")->capture_default_str()->type_name("X")->check(number);
  app.add_option("--c0", options.c0, "c0: auto (1/(2 sigma sqrt(n))) or a finite number > 0")
      ->capture_default_str()
      ->type_name("auto|X")
      ->check([](const std::string& value) -> std::string {
        if (value == "auto") {
          return {};
        }
        const std::optional<float> c0 = to_float(value);
        if (!c0 || !std::isfinite(*c0) || !(*c0 > 0.0f)) {
          return "expected auto or a finite number > 0, got '" + value +
                 "' (0 is not accepted: in the library c0 = 0 means automatic)";
        }
        return {};
      });
  app.add_option("--eval-every", options.eval_every,
                 "bSB: evaluate the signs every K steps (0: only the final configuration)")
      ->capture_default_str()
      ->type_name("K")
      ->transform(decimal);
  app.add_option("--seed", options.seed, "Seed of the initial positions")
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
  app.add_option("--notes", options.notes, "Free text of the notes column")
      ->type_name("TEXT")
      ->check(no_line_break);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e) == kExitOk ? kExitOk : kExitUsage;
  }
  const bool use_dense = dense->count() > 0;
  if (instance->count() == 0 && !use_dense) {
    return usage_error("an instance is required: --instance FILE, or --dense N --instance-seed S");
  }

  try {
    return run(options, use_dense);
  } catch (const std::exception& e) {
    std::cerr << "sbmc_run: " << e.what() << '\n';
    return kExitError;
  }
}
