// The Windows part of sbmc_run (phase 07, part 4, E9): the only code of the project that calls the
// Windows API, compiled only on Windows (apps/CMakeLists.txt). sbmc_run includes this header only
// under _WIN32.
//
// - init_stdout: stdout in binary mode, so that the lines of sbmc_run end with '\n' as on Linux
//   (the CSV files are written in binary mode by the library already).
// - logical_processors: index, core and efficiency class of every logical processor, from
//   GetLogicalProcessorInformationEx(RelationProcessorCore); more than one processor group (more
//   than 64 logical processors) is an error.
// - process_mask and set_process_mask: GetProcessAffinityMask and SetProcessAffinityMask of the
//   current process; the threads created later (the OpenMP team) inherit the mask.
// - computer_name: GetComputerNameA (the value of COMPUTERNAME); windows_version: "Windows
//   <major>.<minor>.<build>" from RtlGetVersion of ntdll.dll (GetVersionEx stops at 6.2 without a
//   manifest).
// - process_environment: the columns cpu_affinity (cpulist of the process mask), cpu_classes
//   (classes of its processors; empty when the processors cannot be read), host and os.
// - apply_affinity: the mask of an --affinity (apps/cpu_topology.hpp) set and read back;
//   std::invalid_argument for a rule that the processor cannot give (e with one class).
// - print_topology: the lines of sbmc_run --print-topology.
// Errors: std::runtime_error with the name of the call and the code of GetLastError.
#pragma once

#ifdef _WIN32

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "cpu_topology.hpp"

namespace sbmc_win {

struct ProcessEnvironment {
  std::string cpu_affinity;
  std::string cpu_classes;
  std::string host;
  std::string os;
};

void init_stdout();
std::vector<sbmc_app::LogicalProcessor> logical_processors();
std::uint64_t process_mask();
void set_process_mask(std::uint64_t mask);
std::string computer_name();
std::string windows_version();
ProcessEnvironment process_environment();
void apply_affinity(const std::string& affinity);
void print_topology(std::ostream& out);

}  // namespace sbmc_win

#endif  // _WIN32
