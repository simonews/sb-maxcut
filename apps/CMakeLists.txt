# Applications, executables in <build>/apps/. Phase 04: sbmc_run, which runs a solver of the
# project on one instance and writes one CSV row per repetition (since phase 07 results schema 2,
# sbmc/results.hpp). Phase 05: sbmc_bruteforce, which runs the exact enumeration C3
# (sbmc/bruteforce.hpp) on one instance and writes one line on stdout, without CSV and without the
# build information. Phase 07: sbmc_spectrum, which estimates the extremes of the spectrum of J of
# one instance (sbmc/spectrum.hpp) and writes them, with the automatic dt and c0, as one line on
# stdout, like sbmc_bruteforce. Phase 07, part 4: on Windows sbmc_run adds win_platform.cpp, the
# only file with Windows API (--affinity, --print-topology and the columns of the process
# environment); cpu_topology.hpp holds the rules of the masks, the same on every platform. Part
# 4b: on Windows the three applications get the runtime DLLs of the toolchain next to them
# (cmake/SbmcWindowsRuntime.cmake), since they are linked dynamically.

# CLI11 (third_party/CLI11, single header): command line parsing, for the applications only. The
# public headers of the project never include it.
add_library(sbmc_cli11 INTERFACE)
add_library(sbmc::cli11 ALIAS sbmc_cli11)
target_include_directories(sbmc_cli11 SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party")

# Build information of the results rows (git hash, uncommitted changes, compiler, build type):
# cmake/sbmc_build_info.cmake writes <build>/generated/sbmc/build_info.hpp at every build, from
# this target, which is always out of date. The header is rewritten only when its content
# changes, and it is declared as a byproduct, so that Ninja recompiles its includers only then.
find_package(Git QUIET)
set(_sbmc_build_info_header "${PROJECT_BINARY_DIR}/generated/sbmc/build_info.hpp")
add_custom_target(sbmc_build_info
  COMMAND "${CMAKE_COMMAND}"
          "-DSBMC_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
          "-DSBMC_OUTPUT=${_sbmc_build_info_header}"
          "-DSBMC_GIT=${GIT_EXECUTABLE}"
          "-DSBMC_COMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
          "-DSBMC_BUILD_TYPE=$<CONFIG>"
          -P "${PROJECT_SOURCE_DIR}/cmake/sbmc_build_info.cmake"
  BYPRODUCTS "${_sbmc_build_info_header}"
  COMMENT "Checking the build information (git hash, uncommitted changes)"
  VERBATIM)

add_executable(sbmc_run sbmc_run.cpp)
if(WIN32)
  # win_platform.cpp calls OpenMP (omp_get_num_procs and the others of --print-topology).
  target_sources(sbmc_run PRIVATE win_platform.cpp)
  target_link_libraries(sbmc_run PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(sbmc_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(sbmc_run sbmc_build_info)
target_include_directories(sbmc_run PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(sbmc_run PRIVATE sbmc::flags sbmc::cpu sbmc::cli11)

add_executable(sbmc_bruteforce sbmc_bruteforce.cpp)
set_target_properties(sbmc_bruteforce PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(sbmc_bruteforce PRIVATE sbmc::flags sbmc::cpu sbmc::cli11)

add_executable(sbmc_spectrum sbmc_spectrum.cpp)
set_target_properties(sbmc_spectrum PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(sbmc_spectrum PRIVATE sbmc::flags sbmc::core sbmc::cli11)

if(WIN32)
  foreach(_sbmc_app IN ITEMS sbmc_run sbmc_bruteforce sbmc_spectrum)
    sbmc_copy_runtime_dlls(${_sbmc_app})
  endforeach()
endif()
