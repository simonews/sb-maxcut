# Tests: doctest unit tests (cpu), Python tests (python), tests of the applications (apps), tool
# tests (tools), CUDA tests and smoke tests (gpu, tile, probe). Folders: core/ (unit tests of
# sbmc::core, executables in <build>/tests/core/, among them T6 on data/gset/ and K2000 on
# data/raw/k2000/), cpu/ (unit tests of sbmc::cpu, executables in <build>/tests/cpu/: C0, C1 with
# T7, the exact enumeration C3 and T2), data/ (input files of the tests), python/ (Python tests,
# run with the interpreter chosen by the root CMakeLists.txt: the reference P0, T8 with traces in
# <build>/tests/python/, the data of data/; and, label apps, sbmc_run with its files in
# <build>/tests/apps/ and sbmc_bruteforce with its files in <build>/tests/bruteforce/), tools/ (test
# tools, executables in <build>/tests/tools/: sbmc_trace), gpu/ (CUDA tests, executables in
# <build>/tests/gpu/), smoke/ (CUDA smoke tests, executables in <build>/tests/smoke/), scripts/
# (tool tests and checks: sync_from_zip.sh, the build information, RPATH). Tests that cannot run
# on the current machine exit with 77, registered as SKIP_RETURN_CODE.

add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party")

# --- CPU ------------------------------------------------------------------------------------
add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity PRIVATE sbmc::flags doctest OpenMP::OpenMP_CXX)
target_compile_definitions(test_sanity PRIVATE "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")
add_test(NAME test_sanity COMMAND test_sanity)
set_tests_properties(test_sanity PROPERTIES LABELS "cpu")

# Unit tests of the core library: one doctest executable per file of core/.
foreach(_sbmc_test IN ITEMS test_philox test_instance test_rudy test_initial_state test_sb
                            test_gset test_k2000 test_results)
  add_executable(${_sbmc_test} core/${_sbmc_test}.cpp)
  set_target_properties(${_sbmc_test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_BINARY_DIR}/core")
  target_link_libraries(${_sbmc_test} PRIVATE sbmc::flags sbmc::core doctest OpenMP::OpenMP_CXX)
  add_test(NAME ${_sbmc_test} COMMAND ${_sbmc_test})
  set_tests_properties(${_sbmc_test} PROPERTIES LABELS "cpu")
endforeach()
target_compile_definitions(test_rudy PRIVATE
                           "SBMC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")
target_compile_definitions(test_sb PRIVATE
                           "SBMC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\""
                           "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")
# T6 (data/gset/) and K2000 (data/raw/k2000/, skipped with 77 until scripts/fetch_data.sh --k2000
# has downloaded it); test_results reads data/best_known.csv.
foreach(_sbmc_test IN ITEMS test_gset test_k2000)
  target_compile_definitions(${_sbmc_test} PRIVATE
                             "SBMC_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data\""
                             "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")
endforeach()
set_tests_properties(test_k2000 PROPERTIES SKIP_RETURN_CODE 77)
target_compile_definitions(test_results PRIVATE "SBMC_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data\"")

# Unit tests of the CPU solvers (sbmc::cpu): the sequential reference C0.
add_executable(test_sb_sequential cpu/test_sb_sequential.cpp)
set_target_properties(test_sb_sequential PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_CURRENT_BINARY_DIR}/cpu")
target_link_libraries(test_sb_sequential PRIVATE sbmc::flags sbmc::cpu doctest OpenMP::OpenMP_CXX)
target_compile_definitions(test_sb_sequential PRIVATE
                           "SBMC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\""
                           "SBMC_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data\""
                           "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")
add_test(NAME test_sb_sequential COMMAND test_sb_sequential)
set_tests_properties(test_sb_sequential PROPERTIES LABELS "cpu")

# C1, simulated bifurcation with OpenMP over blocks of replicas: T7 against C0.
add_executable(test_sb_openmp cpu/test_sb_openmp.cpp)
set_target_properties(test_sb_openmp PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                "${CMAKE_CURRENT_BINARY_DIR}/cpu")
target_link_libraries(test_sb_openmp PRIVATE sbmc::flags sbmc::cpu doctest OpenMP::OpenMP_CXX)
target_compile_definitions(test_sb_openmp PRIVATE
                           "SBMC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\""
                           "SBMC_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data\""
                           "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")
add_test(NAME test_sb_openmp COMMAND test_sb_openmp)
set_tests_properties(test_sb_openmp PROPERTIES LABELS "cpu")

# The exact enumeration C3 (test_bruteforce) and T2, the heuristic solvers against the exact optimum
# (test_t2; phase 05: C0 and C1).
foreach(_sbmc_test IN ITEMS test_bruteforce test_t2)
  add_executable(${_sbmc_test} cpu/${_sbmc_test}.cpp)
  set_target_properties(${_sbmc_test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_BINARY_DIR}/cpu")
  target_link_libraries(${_sbmc_test} PRIVATE sbmc::flags sbmc::cpu doctest OpenMP::OpenMP_CXX)
  target_compile_definitions(${_sbmc_test} PRIVATE
                             "SBMC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")
  add_test(NAME ${_sbmc_test} COMMAND ${_sbmc_test})
  set_tests_properties(${_sbmc_test} PROPERTIES LABELS "cpu")
endforeach()
target_compile_definitions(test_t2 PRIVATE "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")

# --- Test tools -----------------------------------------------------------------------------
# sbmc_trace: runs C0 and writes the trace file read by T8 (python/test_t8.py). Not a test.
add_executable(sbmc_trace tools/sbmc_trace.cpp)
set_target_properties(sbmc_trace PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                            "${CMAKE_CURRENT_BINARY_DIR}/tools")
target_link_libraries(sbmc_trace PRIVATE sbmc::flags sbmc::cpu)

# --- Python ---------------------------------------------------------------------------------
# Scripts of python/, run with the interpreter of the root CMakeLists.txt in isolated mode and
# without bytecode, so that nothing is written into the source tree. Without an interpreter the
# tests are registered anyway and exit with 77.
#   sbmc_add_python_test(<name> <label> [WORKING_DIRECTORY <dir>] ARGS <arguments>...)
function(sbmc_add_python_test name label)
  cmake_parse_arguments(PARSE_ARGV 2 _sbmc "" "WORKING_DIRECTORY" "ARGS")
  if(_sbmc_python)
    set(_sbmc_command "${_sbmc_python}" -I -B "${CMAKE_CURRENT_SOURCE_DIR}/python/${name}.py"
                      ${_sbmc_ARGS})
  else()
    set(_sbmc_command bash -c "echo '${name}: no Python interpreter: run scripts/setup_python.sh, then configure again' && exit 77")
  endif()
  if(_sbmc_WORKING_DIRECTORY)
    add_test(NAME ${name} COMMAND ${_sbmc_command} WORKING_DIRECTORY "${_sbmc_WORKING_DIRECTORY}")
  else()
    add_test(NAME ${name} COMMAND ${_sbmc_command})
  endif()
  set_tests_properties(${name} PROPERTIES LABELS "${label}" SKIP_RETURN_CODE 77)
endfunction()

# P0 alone (test_p0) and T8, C0 against P0 through the traces of sbmc_trace, written to
# <build>/tests/python/ (test_t8).
set(_sbmc_python_common
    "--pyroot" "${PROJECT_SOURCE_DIR}/python" "--data" "${CMAKE_CURRENT_SOURCE_DIR}/data")
sbmc_add_python_test(test_p0 python ARGS ${_sbmc_python_common})
sbmc_add_python_test(test_t8 python ARGS ${_sbmc_python_common}
                     "--tool" "$<TARGET_FILE:sbmc_trace>"
                     "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/python")
# The data of data/: SHA256SUMS and the 13 G-set files, read by P0, and best_known.csv.
sbmc_add_python_test(test_data python
                     ARGS "--pyroot" "${PROJECT_SOURCE_DIR}/python"
                          "--data-root" "${PROJECT_SOURCE_DIR}/data")

# --- Applications ---------------------------------------------------------------------------
# sbmc_run (apps/), run from the repository root (its defaults are relative to it); its CSV files
# go to <build>/tests/apps/; --strict-fp tells the strict presets, where C1 and C0 must give the
# same rows.
sbmc_add_python_test(test_sbmc_run apps WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                     ARGS "--sbmc-run" "$<TARGET_FILE:sbmc_run>"
                          "--trace-tool" "$<TARGET_FILE:sbmc_trace>"
                          "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/apps"
                          "--strict-fp" "$<BOOL:${SBMC_STRICT_FP}>")
# sbmc_bruteforce (apps/), with absolute paths; its files go to <build>/tests/bruteforce/.
sbmc_add_python_test(test_sbmc_bruteforce apps
                     ARGS "--tool" "$<TARGET_FILE:sbmc_bruteforce>"
                          "--data" "${CMAKE_CURRENT_SOURCE_DIR}/data"
                          "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/bruteforce")

# --- Tools ----------------------------------------------------------------------------------
add_test(NAME test_sync_from_zip
         COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_sync_from_zip.sh"
                 "${PROJECT_SOURCE_DIR}/scripts/sync_from_zip.sh")
set_tests_properties(test_sync_from_zip PROPERTIES LABELS "tools" SKIP_RETURN_CODE 77)
# cmake/sbmc_build_info.cmake in temporary git repositories: dirty ignores results/ of the root.
add_test(NAME test_build_info
         COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_build_info.sh" "${CMAKE_COMMAND}"
                 "${PROJECT_SOURCE_DIR}/cmake/sbmc_build_info.cmake")
set_tests_properties(test_build_info PROPERTIES LABELS "tools" SKIP_RETURN_CODE 77)

# --- GPU smoke tests ------------------------------------------------------------------------
# Executables in <build>/tests/smoke/, next to their sources' layout (scripts/env_report.sh runs
# <build>/tests/smoke/smoke_simt). Every one of them loads CUDA libraries, so it links
# sbmc::cuda_rpath (cmake/SbmcFlags.cmake) and is listed in _sbmc_cuda_executables, the files
# checked by test_cuda_rpath.
if(SBMC_ENABLE_CUDA)
  add_executable(smoke_simt smoke/smoke_simt.cu)
  set_target_properties(smoke_simt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              "${CMAKE_CURRENT_BINARY_DIR}/smoke")
  target_link_libraries(smoke_simt PRIVATE sbmc::flags sbmc::cuda_rpath CUDA::cudart CUDA::cublas
                                           CUDA::cublasLt)
  add_test(NAME smoke_simt COMMAND smoke_simt)
  set_tests_properties(smoke_simt PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77)
  set(_sbmc_cuda_executables "$<TARGET_FILE:smoke_simt>")

  if(SBMC_ENABLE_TILE)
    # Tile targets: sm_89 only, whatever CMAKE_CUDA_ARCHITECTURES says.
    set(_sbmc_tile_options "$<$<COMPILE_LANGUAGE:CUDA>:-enable-tile;-std=c++20>")

    add_executable(smoke_tile smoke/smoke_tile.cu)
    set_target_properties(smoke_tile PROPERTIES CUDA_ARCHITECTURES 89 RUNTIME_OUTPUT_DIRECTORY
                                                "${CMAKE_CURRENT_BINARY_DIR}/smoke")
    target_compile_options(smoke_tile PRIVATE "${_sbmc_tile_options}")
    target_link_libraries(smoke_tile PRIVATE sbmc::flags sbmc::cuda_rpath CUDA::cudart)
    add_test(NAME smoke_tile COMMAND smoke_tile)
    set_tests_properties(smoke_tile PROPERTIES LABELS "gpu;tile" SKIP_RETURN_CODE 77)
    list(APPEND _sbmc_cuda_executables "$<TARGET_FILE:smoke_tile>")

    # FMA probes: one source, two builds. They do not link sbmc::flags: --fmad is set here, and
    # the host reference is never contracted (-ffp-contract=off in both builds). They link
    # sbmc::cuda_rpath, a linker option that leaves the compilation unchanged.
    foreach(_sbmc_variant IN ITEMS on off)
      if(_sbmc_variant STREQUAL "on")
        set(_sbmc_fmad "--fmad=true")
        set(_sbmc_fmad_on 1)
      else()
        set(_sbmc_fmad "--fmad=false")
        set(_sbmc_fmad_on 0)
      endif()
      set(_sbmc_probe "fma_probe_tile_fmad_${_sbmc_variant}")
      add_executable(${_sbmc_probe} smoke/fma_probe_tile.cu)
      set_target_properties(${_sbmc_probe} PROPERTIES CUDA_ARCHITECTURES 89 RUNTIME_OUTPUT_DIRECTORY
                                                      "${CMAKE_CURRENT_BINARY_DIR}/smoke")
      target_compile_options(${_sbmc_probe} PRIVATE
        "${_sbmc_tile_options}"
        "$<$<COMPILE_LANGUAGE:CUDA>:${_sbmc_fmad};-Xcompiler=-ffp-contract=off>")
      target_compile_definitions(${_sbmc_probe} PRIVATE "SBMC_PROBE_FMAD_ON=${_sbmc_fmad_on}")
      target_link_libraries(${_sbmc_probe} PRIVATE sbmc::cuda_rpath CUDA::cudart)
      add_test(NAME ${_sbmc_probe} COMMAND ${_sbmc_probe})
      set_tests_properties(${_sbmc_probe} PROPERTIES LABELS "gpu;tile;probe" SKIP_RETURN_CODE 77)
      list(APPEND _sbmc_cuda_executables "$<TARGET_FILE:${_sbmc_probe}>")
    endforeach()
  endif()

  # T11 on the GPU: cuRAND, the project's Philox and x0 on the device against the host. Built
  # for every architecture of the preset. The device API of cuRAND is header-only
  # (curand_kernel.h): no cuRAND library is linked.
  add_executable(test_curand_philox gpu/test_curand_philox.cu)
  set_target_properties(test_curand_philox PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                      "${CMAKE_CURRENT_BINARY_DIR}/gpu")
  target_link_libraries(test_curand_philox PRIVATE sbmc::flags sbmc::cuda_rpath sbmc::headers
                                                   CUDA::cudart)
  add_test(NAME test_curand_philox COMMAND test_curand_philox)
  set_tests_properties(test_curand_philox PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77)
  list(APPEND _sbmc_cuda_executables "$<TARGET_FILE:test_curand_philox>")

  # Every CUDA executable of the build must find the CUDA libraries through RPATH (a folder with
  # libcudart.so.13) and have no RUNPATH. Skipped (77) when readelf is missing.
  add_test(NAME test_cuda_rpath
           COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_rpath.sh"
                   ${_sbmc_cuda_executables})
  set_tests_properties(test_cuda_rpath PROPERTIES LABELS "tools" SKIP_RETURN_CODE 77)
endif()
