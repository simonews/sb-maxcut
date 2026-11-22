// sbmc_spectrum: estimate of the extremes of the spectrum of J of one instance with the Lanczos
// method (sbmc/spectrum.hpp), with the automatic dt and the two automatic c0 derived from it,
// written as one line on stdout.
//
//   sbmc_spectrum (--instance FILE | --dense N --instance-seed S) [--name NAME] [--iterations K]
//
// Run: the instance is read, then estimate_spectrum runs K Lanczos steps (default 100,
// kSpectrumIterations; at most 100000; min(K, n) are done). Output, the only line on stdout, with
// the fields separated by one space:
//   SPECTRUM instance=<name> n=<n> m=<m> lambda_min=<x> lambda_max=<x> iterations=<k>
//            restarts=<r> dt_auto=<f> c0_lambda=<f> c0_sigma=<f> t_ms=<t>
// m is the number of edges of a rudy graph and n(n - 1)/2 for a dense instance; lambda_min and
// lambda_max are doubles in the shortest form that reads back equal; dt_auto = spectral_dt (the
// factor 1), c0_lambda = spectral_c0 and c0_sigma = auto_c0 are floats in the shortest form, or
// "none" when the function throws InvalidArgument (for example on a graph without edges); t_ms is
// the duration of estimate_spectrum, with three decimals.
//
// Exit status: 0 on success; 1 for errors of the library (for example an invalid instance) and of
// reading; 2 for errors of the command line, with a message on stderr. A name, given with --name
// or taken from the file name, may not be empty or contain a space, a tab, '\r' or '\n' (exit
// status 2): the name is one field of the output line. Every message on stderr starts with
// "sbmc_spectrum: ".
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <CLI11/CLI11.hpp>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sb.hpp>
#include <sbmc/spectrum.hpp>

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
  int iterations = sbmc::kSpectrumIterations;
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

// The shortest form that reads back as the same value (double or float).
template <typename T>
std::string shortest(T value) {
  std::array<char, 64> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("nan");
}

// The value of a function of the library in the shortest form, or "none" when it throws
// InvalidArgument (for example spectral_dt on a graph without edges).
std::string float_or_none(const std::function<float()>& compute) {
  try {
    return shortest(compute());
  } catch (const sbmc::InvalidArgument&) {
    return "none";
  }
}

// Copied from sbmc_bruteforce.cpp: the output line separates its fields with spaces, so a name
// may not be empty or contain a space, a tab, '\r' or '\n'. Returns the reason, or an empty
// string.
std::string name_problem(std::string_view name) {
  if (name.empty()) {
    return "the name cannot be empty";
  }
  if (name.find_first_of(" \t\r\n") != std::string_view::npos) {
    return "the name cannot contain spaces, tabs or line breaks (one field of the output line)";
  }
  return {};
}

int usage_error(std::string_view message) {
  std::cerr << "sbmc_spectrum: " << message << "\nRun with --help for more information.\n";
  return kExitUsage;
}

int run(const Options& options, bool dense, bool named) {
  // 1. Instance.
  sbmc::Instance instance;
  std::int64_t edges = 0;
  if (dense) {
    instance = sbmc::DenseSpinGlass{options.dense_n, options.instance_seed};
  } else {
    sbmc::Graph graph = sbmc::read_rudy(options.instance_file);
    edges = static_cast<std::int64_t>(graph.edges.size());
    instance = std::move(graph);
  }

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

  // 2. The estimate (all the checks of the instance are in the library), timed.
  const auto start = std::chrono::steady_clock::now();
  const sbmc::SpectrumEstimate s = sbmc::estimate_spectrum(instance, options.iterations);
  const double t_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  // 3. The values derived from it, and the output line. estimate_spectrum has checked n, so
  // n (n - 1) / 2 of a dense instance (n <= 32768) cannot overflow.
  const std::int64_t n = sbmc::num_spins(instance);
  const std::int64_t m = dense ? n * (n - 1) / 2 : edges;
  const std::string dt_auto = float_or_none([&] { return sbmc::spectral_dt(s); });
  const std::string c0_lambda = float_or_none([&] { return sbmc::spectral_c0(s); });
  const std::string c0_sigma = float_or_none([&] { return sbmc::auto_c0(instance); });
  std::cout << "SPECTRUM instance=" << name << " n=" << n << " m=" << m
            << " lambda_min=" << shortest(s.lambda_min) << " lambda_max=" << shortest(s.lambda_max)
            << " iterations=" << s.iterations << " restarts=" << s.restarts
            << " dt_auto=" << dt_auto << " c0_lambda=" << c0_lambda << " c0_sigma=" << c0_sigma
            << " t_ms=" << fixed3(t_ms) << '\n'
            << std::flush;
  return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  CLI::App app{
      "Estimate of the extremes of the spectrum of J of one instance (Lanczos method), with the "
      "automatic dt and c0 derived from it; writes one line on stdout.",
      "sbmc_spectrum"};
  app.failure_message([](const CLI::App* parser, const CLI::Error& e) {
    const std::string message = CLI::FailureMessage::simple(parser, e);
    return message.starts_with("sbmc_spectrum: ") ? message : "sbmc_spectrum: " + message;
  });
  app.footer(
      "Output: SPECTRUM instance=<name> n=<n> m=<m> lambda_min=<x> lambda_max=<x> "
      "iterations=<k> restarts=<r> dt_auto=<f> c0_lambda=<f> c0_sigma=<f> t_ms=<t>. dt_auto = "
      "1.25 sqrt(2 / (1 - lambda_min / lambda_max)), c0_lambda = 1 / lambda_max, c0_sigma = "
      "1 / (2 sigma sqrt(n)); \"none\" when a value is not defined (for example without edges). "
      "Exit status: 0 success; 1 error of the library or of reading; 2 error of the command "
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
  app.add_option("--iterations", options.iterations,
                 "Lanczos steps (1 to 100000; min(K, n) are done)")
      ->capture_default_str()
      ->type_name("K")
      ->transform(decimal)
      ->check(CLI::Range(1, 100000));

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
    std::cerr << "sbmc_spectrum: " << e.what() << '\n';
    return kExitError;
  }
}
