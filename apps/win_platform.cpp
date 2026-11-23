// The Windows part of sbmc_run: see win_platform.hpp. Compiled only on Windows.
#include "win_platform.hpp"

#ifndef _WIN32
#error "win_platform.cpp is compiled only on Windows (apps/CMakeLists.txt)"
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <fcntl.h>
#include <io.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sbmc_win {
namespace {

std::runtime_error api_error(const std::string& call) {
  return std::runtime_error(call + " failed (GetLastError " + std::to_string(GetLastError()) + ")");
}

// The layout of PROCESSOR_RELATIONSHIP: BYTE Flags; BYTE EfficiencyClass; BYTE Reserved[20];
// WORD GroupCount; GROUP_AFFINITY GroupMask[]. Older headers of mingw-w64 name the second byte
// Reserved[0]: it is read by its offset.
constexpr std::size_t kEfficiencyClassOffset = 1;
static_assert(offsetof(PROCESSOR_RELATIONSHIP, GroupCount) == 22,
              "unexpected layout of PROCESSOR_RELATIONSHIP");

}  // namespace

void init_stdout() {
  std::fflush(stdout);
  _setmode(_fileno(stdout), _O_BINARY);
}

std::vector<sbmc_app::LogicalProcessor> logical_processors() {
  if (GetActiveProcessorGroupCount() != 1) {
    throw std::runtime_error(std::to_string(GetActiveProcessorGroupCount()) +
                             " processor groups: more than 64 logical processors are not "
                             "supported");
  }
  DWORD length = 0;
  if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    throw api_error("GetLogicalProcessorInformationEx (size)");
  }
  std::vector<unsigned char> buffer(length);
  auto* const first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &length)) {
    throw api_error("GetLogicalProcessorInformationEx");
  }
  std::vector<sbmc_app::LogicalProcessor> processors;
  int core = 0;
  for (DWORD offset = 0; offset < length;) {
    const auto* const info =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
    if (info->Size == 0) {
      throw std::runtime_error("GetLogicalProcessorInformationEx: an entry of size 0");
    }
    if (info->Relationship == RelationProcessorCore) {
      const PROCESSOR_RELATIONSHIP& relation = info->Processor;
      const int efficiency_class =
          reinterpret_cast<const unsigned char*>(&relation)[kEfficiencyClassOffset];
      if (relation.GroupCount != 1 || relation.GroupMask[0].Group != 0) {
        throw std::runtime_error(
            "a core outside processor group 0: more than 64 logical "
            "processors are not supported");
      }
      const std::uint64_t mask = relation.GroupMask[0].Mask;
      for (int bit = 0; bit < 64; ++bit) {
        if (((mask >> bit) & 1U) != 0) {
          processors.push_back({bit, core, efficiency_class});
        }
      }
      ++core;
    }
    offset += info->Size;
  }
  sbmc_app::check_processors(processors);
  return processors;
}

std::uint64_t process_mask() {
  DWORD_PTR process = 0;
  DWORD_PTR system = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) {
    throw api_error("GetProcessAffinityMask");
  }
  return static_cast<std::uint64_t>(process);
}

void set_process_mask(std::uint64_t mask) {
  if (!SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(mask))) {
    throw api_error("SetProcessAffinityMask");
  }
}

std::string computer_name() {
  char name[MAX_COMPUTERNAME_LENGTH + 1] = {};
  DWORD size = sizeof(name);
  if (!GetComputerNameA(name, &size)) {
    return "unknown";
  }
  return std::string(name, size);
}

std::string windows_version() {
  using RtlGetVersionFunction = LONG(WINAPI*)(OSVERSIONINFOW*);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  if (ntdll == nullptr) {
    return "Windows unknown";
  }
  const auto rtl_get_version = reinterpret_cast<RtlGetVersionFunction>(
      reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")));
  if (rtl_get_version == nullptr) {
    return "Windows unknown";
  }
  OSVERSIONINFOW info{};
  info.dwOSVersionInfoSize = sizeof(info);
  if (rtl_get_version(&info) != 0) {
    return "Windows unknown";
  }
  return "Windows " + std::to_string(info.dwMajorVersion) + "." +
         std::to_string(info.dwMinorVersion) + "." + std::to_string(info.dwBuildNumber);
}

ProcessEnvironment process_environment() {
  ProcessEnvironment environment;
  const std::uint64_t mask = process_mask();
  environment.cpu_affinity = sbmc_app::cpulist(mask);
  try {
    environment.cpu_classes = sbmc_app::cpu_classes(logical_processors(), mask);
  } catch (const std::exception&) {
    environment.cpu_classes.clear();  // the column stays empty when the processors are unknown
  }
  environment.host = computer_name();
  environment.os = windows_version();
  return environment;
}

void apply_affinity(const std::string& affinity) {
  const std::optional<sbmc_app::Affinity> rule = sbmc_app::parse_affinity(affinity);
  if (!rule) {
    throw std::invalid_argument("--affinity " + affinity + ": expected p1, p, e or all");
  }
  const std::uint64_t mask = sbmc_app::affinity_mask(logical_processors(), *rule);
  set_process_mask(mask);
  const std::uint64_t effective = process_mask();
  if (effective != mask) {
    throw std::runtime_error("the process mask is " + sbmc_app::cpulist(effective) +
                             " after setting " + sbmc_app::cpulist(mask));
  }
}

void print_topology(std::ostream& out) {
  const std::vector<sbmc_app::LogicalProcessor> processors = logical_processors();
  int cores = 0;
  for (const sbmc_app::LogicalProcessor& p : processors) {
    cores = std::max(cores, p.core + 1);
  }
  out << "TOPOLOGY logical=" << processors.size() << " cores=" << cores
      << " classes=" << sbmc_app::class_count(processors) << '\n';
  for (const sbmc_app::LogicalProcessor& p : processors) {
    out << "CPU " << p.index << " core=" << p.core << " class=" << p.efficiency_class << '\n';
  }
  std::map<int, std::uint64_t, std::greater<>> class_masks;
  for (const sbmc_app::LogicalProcessor& p : processors) {
    class_masks[p.efficiency_class] |= std::uint64_t{1} << p.index;
  }
  for (const auto& [efficiency_class, class_mask] : class_masks) {
    out << "CLASS " << efficiency_class << " count=" << sbmc_app::mask_count(class_mask)
        << " cpus=" << sbmc_app::cpulist(class_mask) << '\n';
  }
  for (const sbmc_app::Affinity rule : {sbmc_app::Affinity::p1, sbmc_app::Affinity::p,
                                        sbmc_app::Affinity::e, sbmc_app::Affinity::all}) {
    out << "AFFINITY " << sbmc_app::affinity_name(rule);
    try {
      const std::uint64_t mask = sbmc_app::affinity_mask(processors, rule);
      out << " count=" << sbmc_app::mask_count(mask) << " cpus=" << sbmc_app::cpulist(mask) << '\n';
    } catch (const std::invalid_argument&) {
      out << " unavailable (one efficiency class)\n";
    }
  }
  const std::uint64_t mask = process_mask();
  out << "MASK count=" << sbmc_app::mask_count(mask) << " cpus=" << sbmc_app::cpulist(mask)
      << " classes=" << sbmc_app::cpu_classes(processors, mask) << '\n';
  const char* bind = "unknown";
  switch (omp_get_proc_bind()) {
    case omp_proc_bind_false:
      bind = "false";
      break;
    case omp_proc_bind_true:
      bind = "true";
      break;
    case omp_proc_bind_master:
      bind = "master";
      break;
    case omp_proc_bind_close:
      bind = "close";
      break;
    case omp_proc_bind_spread:
      bind = "spread";
      break;
  }
  out << "OPENMP num_procs=" << omp_get_num_procs() << " max_threads=" << omp_get_max_threads()
      << " proc_bind=" << bind << '\n'
      << std::flush;
}

}  // namespace sbmc_win
