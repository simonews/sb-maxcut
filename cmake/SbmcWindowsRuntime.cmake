# Windows only (phase 07, part 4b, decision W5): the executables are linked dynamically with the
# MinGW-w64 GCC of CLion, and the runtime DLLs of the toolchain are copied next to every
# executable started outside ctest, so that it starts without the folders of CLion in PATH (from
# WSL through the interop, or from a clean PowerShell). Included by the top-level CMakeLists.txt
# when WIN32.
#
# Why not static: with the MinGW-w64 of CLion (g++ 15.2.0) "g++ -fopenmp -static" fails at link
# time (__intrinsic_setjmpex from libpthread.a and __ms_vsnprintf from libmsvcrt.a are undefined),
# so FindOpenMP does not find OpenMP; the dynamic link of the same program works.
#
# At configure time: the toolchain must hold the four runtime DLLs below, looked up next to the
# compiler (the layout of CLion: bin/ holds g++.exe and the DLLs) and then with
# "<compiler> -print-file-name=<dll>" (the layout of the Debian cross compilers used in the
# sandbox); a missing DLL is a configure error that names it. CMAKE_OBJDUMP (found by CMake next
# to the compiler) must exist. The folders of the DLLs and objdump go to
# <build>/sbmc_runtime_config.cmake.
#
# sbmc_copy_runtime_dlls(<target>): after every link of <target>, cmake/sbmc_copy_runtime.cmake
# reads the imports of the executable with objdump -p and copies every imported DLL found in
# those folders next to it, then does the same for the imports of the copied DLLs. The list is
# the one of the imports, not a fixed one; a DLL found in none of the folders is a DLL of the
# system (KERNEL32.dll, msvcrt.dll) and stays where it is. scripts/windows/build.ps1 checks the
# result: the imported DLLs that are not in System32 are next to the executable, and
# sbmc_run.exe --version runs with a PATH without CLion.
#
# Part 4d: the copies of different executables write the same files (sbmc_run, sbmc_bruteforce
# and sbmc_spectrum share apps/), and on Windows a file being written by one copy cannot be
# opened by another: on the laptop two of them ran in parallel and the build of win-cpu-strict
# failed ("file COPY_FILE failed to copy"). In the Ninja generator the POST_BUILD command is part
# of the link step, so the links of the targets that copy go to the job pool sbmc_runtime_copy of
# depth 1: one link and its copy at a time, the compilations still in parallel. Other generators
# ignore job pools; the presets use Ninja.
include_guard(GLOBAL)

if(NOT WIN32)
  return()
endif()

if(NOT CMAKE_OBJDUMP OR NOT EXISTS "${CMAKE_OBJDUMP}")
  message(FATAL_ERROR
    "objdump of the toolchain not found (CMAKE_OBJDUMP='${CMAKE_OBJDUMP}'): it is needed to copy "
    "the runtime DLLs next to the executables")
endif()

get_filename_component(_sbmc_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
set(SBMC_RUNTIME_DLLS libgomp-1.dll libwinpthread-1.dll libstdc++-6.dll libgcc_s_seh-1.dll)
set(SBMC_RUNTIME_DIRS "${_sbmc_compiler_dir}")
foreach(_sbmc_dll IN LISTS SBMC_RUNTIME_DLLS)
  if(EXISTS "${_sbmc_compiler_dir}/${_sbmc_dll}")
    continue()
  endif()
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${_sbmc_dll}"
    OUTPUT_VARIABLE _sbmc_dll_path
    ERROR_QUIET
    RESULT_VARIABLE _sbmc_dll_status
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_sbmc_dll_status EQUAL 0 AND IS_ABSOLUTE "${_sbmc_dll_path}" AND EXISTS "${_sbmc_dll_path}")
    file(REAL_PATH "${_sbmc_dll_path}" _sbmc_dll_path)
    get_filename_component(_sbmc_dll_dir "${_sbmc_dll_path}" DIRECTORY)
    list(APPEND SBMC_RUNTIME_DIRS "${_sbmc_dll_dir}")
  else()
    message(FATAL_ERROR
      "runtime DLL ${_sbmc_dll} not found in the toolchain: not next to the compiler "
      "(${_sbmc_compiler_dir}) and not given by ${CMAKE_CXX_COMPILER} -print-file-name")
  endif()
endforeach()
list(REMOVE_DUPLICATES SBMC_RUNTIME_DIRS)
message(STATUS "Runtime DLLs (${SBMC_RUNTIME_DLLS}) found in: ${SBMC_RUNTIME_DIRS}")

set(SBMC_RUNTIME_CONFIG "${PROJECT_BINARY_DIR}/sbmc_runtime_config.cmake")
set(_sbmc_config_text "# Written at configure time by cmake/SbmcWindowsRuntime.cmake.\n")
string(APPEND _sbmc_config_text "set(SBMC_OBJDUMP [==[${CMAKE_OBJDUMP}]==])\n")
string(APPEND _sbmc_config_text "set(SBMC_RUNTIME_DIRS \"\")\n")
foreach(_sbmc_dir IN LISTS SBMC_RUNTIME_DIRS)
  string(APPEND _sbmc_config_text "list(APPEND SBMC_RUNTIME_DIRS [==[${_sbmc_dir}]==])\n")
endforeach()
file(CONFIGURE OUTPUT "${SBMC_RUNTIME_CONFIG}" CONTENT "${_sbmc_config_text}" @ONLY)

set_property(GLOBAL APPEND PROPERTY JOB_POOLS sbmc_runtime_copy=1)

function(sbmc_copy_runtime_dlls target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
            "-DSBMC_EXE=$<TARGET_FILE:${target}>"
            "-DSBMC_RUNTIME_CONFIG=${SBMC_RUNTIME_CONFIG}"
            -P "${PROJECT_SOURCE_DIR}/cmake/sbmc_copy_runtime.cmake"
    VERBATIM)
  set_property(TARGET ${target} PROPERTY JOB_POOL_LINK sbmc_runtime_copy)
endfunction()
