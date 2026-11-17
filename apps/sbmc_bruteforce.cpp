// sbmc_bruteforce: exact Max-Cut of one instance by Gray-code enumeration (C3,
// sbmc/bruteforce.hpp), written as one line on stdout.
//
//   sbmc_bruteforce (--instance FILE | --dense N --instance-seed S) [--name NAME] [--threads T]
//
// Run: the instance is read (t_setup_ms), then solve_bruteforce runs with T threads (t_solve_ms;
// T = 0, the default, means the default number of OpenMP threads). Output, the only line on
// stdout, with the fields separated by one space:
//   BRUTEFORCE instance=<name> n=<n> m=<m> W=<W> max_cut=<C> min_energy=<E> optimal_count=<k>
//              threads=<t> t_setup_ms=<x> t_solve_ms=<y> config=<s>
// m is the number of edges of a rudy graph and n(n - 1)/2 for a dense instance; the times are in
// fixed notation with three decimals; config is the canonical optimum, one '+' or '-' per spin.
//
// Exit status: 0 on success; 1 for errors of the library (for example n > 40 or an invalid
// instance) and of reading; 2 for errors of the command line, with a message on stderr. A name,
// given with --name or taken from the file name, may not be empty or contain a space, a tab,
// '\r' or '\n' (exit status 2): the name is one field of the output line. Every message on
// stderr starts with "sbmc_bruteforce: ".
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <CLI11/CLI11.hpp>

#include <sbmc/bruteforce.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>

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
  int threads = 0;
};

// Copied from sbmc_run.cpp. CLI11 converts integers with strtoll and strtoull in base 0 ("010"
// is octal, "0x10" hexadecimal) and also accepts digit separators and "true": the integer options
// take decimal integers only. This transform rejects anything else and removes leading zeros, so
// that CLI11 reads the value in base 10; CLI11 still checks the range of the destination type.
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

// Copied from sbmc_run.cpp: fixed notation with three decimals.
std::string fixed3(double value) {
  std::array<char, 512> text{};
  const auto [end, error] =
      std::to_chars(text.data(), text.data() + text.size(), value, std::chars_format::fixed, 3);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("nan");
}

// The output line separates its fields with spaces: a name may not be empty or contain one of
// the whitespace characters space, tab, '\r', '\n'. Returns the reason, or an empty string.
std::string name_problem(std::string_view name) {
  if (name.empty()) {
    return "the name cannot be empty";
  }
  if (name.find_first_of(" \t\r\n") != std::string_view::npos) {
    return "the name cannot contain spaces, tabs or line breaks (one field of the output line)";
  }
  return {};
}

std::string config_text(const sbmc::BruteForceResult& result) {
  std::string text;
  text.reserve(result.best_spins.size());
  for (const sbmc::Spin s : result.best_spins) {
    text.push_back(s > 0 ? '+' : '-');
  }
  return text;
}

int usage_error(std::string_view message) {
  std::cerr << "sbmc_bruteforce: " << message << "\nRun with --help for more information.\n";
  return kExitUsage;
}

int run(const Options& options, bool dense, bool named) {
  // 1. Instance; t_setup_ms is the time of this phase.
  const auto setup_start = std::chrono::steady_clock::now();
  sbmc::Instance instance;
  std::int64_t edges = 0;
  if (dense) {
    instance = sbmc::DenseSpinGlass{options.dense_n, options.instance_seed};
  } else {
    sbmc::Graph graph = sbmc::read_rudy(options.instance_file);
    edges = static_cast<std::int64_t>(graph.edges.size());
    instance = std::move(graph);
  }
  const double t_setup_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start)
          .count();

  // The default name, checked like --name once the file has been read (a file name with a space
  // would break the output line).
  std::string name = options.name;
  if (!named) {
    name = dense ? "dense_n" + std::to_string(options.dense_n) + "_s" +
                       std::to_string(options.instance_seed)
                 : std::filesystem::path(options.instance_file).stem().string();
    if (const std::string problem = name_problem(name); !problem.empty()) {
      return usage_error("the default name '" + name + "' is not valid: " + problem +
                         "; choose one with --name");
    }
  }

  // 2. Exact solution (all the checks of the instance are in the library).
  const auto solve_start = std::chrono::steady_clock::now();
  const sbmc::BruteForceResult result = sbmc::solve_bruteforce(instance, options.threads);
  const double t_solve_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solve_start)
          .count();

  // 3. The output line. A dense instance has n <= 40 here: n (n - 1) / 2 cannot overflow.
  const std::int64_t m = dense ? result.n * (result.n - 1) / 2 : edges;
  std::cout << "BRUTEFORCE instance=" << name << " n=" << result.n << " m=" << m
            << " W=" << result.total_weight << " max_cut=" << result.max_cut
            << " min_energy=" << result.min_energy << " optimal_count=" << result.optimal_count
            << " threads=" << result.threads << " t_setup_ms=" << fixed3(t_setup_ms)
            << " t_solve_ms=" << fixed3(t_solve_ms) << " config=" << config_text(result) << '\n'
            << std::flush;
  return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  CLI::App app{
      "Exact Max-Cut of one instance (n <= 40) by Gray-code enumeration (C3), parallel with "
      "OpenMP; writes one line on stdout.",
      "sbmc_bruteforce"};
  app.failure_message([](const CLI::App* parser, const CLI::Error& e) {
    const std::string message = CLI::FailureMessage::simple(parser, e);
    return message.starts_with("sbmc_bruteforce: ") ? message : "sbmc_bruteforce: " + message;
  });
  app.footer(
      "Output: BRUTEFORCE instance=<name> n=<n> m=<m> W=<W> max_cut=<C> min_energy=<E> "
      "optimal_count=<k> threads=<t> t_setup_ms=<x> t_solve_ms=<y> config=<s>. Exit status: 0 "
      "success; 1 error of the library (for example n > 40) or of reading; 2 error of the command "
      "line.");

  const CLI::Validator decimal = decimal_integer();

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
  CLI::Option* name =
      app.add_option("--name", options.name,
                     "Instance name in the output line (default: the file name without folder "
                     "and extension, or dense_n<N>_s<S>); no spaces, tabs or line breaks")
          ->check([](const std::string& value) { return name_problem(value); });
  app.add_option("--threads", options.threads,
                 "OpenMP threads of the enumeration (0: the default number of OpenMP)")
      ->capture_default_str()
      ->type_name("T")
      ->transform(decimal)
      ->check(CLI::Range(0, std::numeric_limits<int>::max()));

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
    return run(options, use_dense, name->count() > 0);
  } catch (const std::exception& e) {
    std::cerr << "sbmc_bruteforce: " << e.what() << '\n';
    return kExitError;
  }
}
