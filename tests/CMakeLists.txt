# Tests: doctest unit tests (cpu), tool tests (tools), CUDA tests and smoke tests (gpu, tile,
# probe). Folders: core/ (unit tests of sbmc::core, executables in <build>/tests/core/), data/
# (input files of the tests), gpu/ (CUDA tests, executables in <build>/tests/gpu/), smoke/ (CUDA
# smoke tests, executables in <build>/tests/smoke/), scripts/ (tool tests and checks).
# Tests that cannot run on the current machine exit with 77, registered as SKIP_RETURN_CODE.

add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party")

# --- CPU ------------------------------------------------------------------------------------
add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity PRIVATE sbmc::flags doctest OpenMP::OpenMP_CXX)
target_compile_definitions(test_sanity PRIVATE "SBMC_STRICT_FP_BUILD=$<BOOL:${SBMC_STRICT_FP}>")
add_test(NAME test_sanity COMMAND test_sanity)
set_tests_properties(test_sanity PROPERTIES LABELS "cpu")

# Unit tests of the core library: one doctest executable per file of core/.
foreach(_sbmc_test IN ITEMS test_philox test_instance test_rudy test_initial_state)
  add_executable(${_sbmc_test} core/${_sbmc_test}.cpp)
  set_target_properties(${_sbmc_test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_BINARY_DIR}/core")
  target_link_libraries(${_sbmc_test} PRIVATE sbmc::flags sbmc::core doctest OpenMP::OpenMP_CXX)
  add_test(NAME ${_sbmc_test} COMMAND ${_sbmc_test})
  set_tests_properties(${_sbmc_test} PROPERTIES LABELS "cpu")
endforeach()
target_compile_definitions(test_rudy PRIVATE
                           "SBMC_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

# --- Tools ----------------------------------------------------------------------------------
add_test(NAME test_sync_from_zip
         COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_sync_from_zip.sh"
                 "${PROJECT_SOURCE_DIR}/scripts/sync_from_zip.sh")
set_tests_properties(test_sync_from_zip PROPERTIES LABELS "tools" SKIP_RETURN_CODE 77)

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
