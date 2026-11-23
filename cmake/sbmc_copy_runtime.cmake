# Copies the runtime DLLs of the toolchain next to a Windows executable (phase 07, part 4b,
# decision W5). Run with cmake -P after every link of the executable, by the POST_BUILD command
# of sbmc_copy_runtime_dlls() (cmake/SbmcWindowsRuntime.cmake):
#
#   cmake -DSBMC_EXE=<executable> -DSBMC_RUNTIME_CONFIG=<build>/sbmc_runtime_config.cmake
#         -P cmake/sbmc_copy_runtime.cmake
#
# SBMC_RUNTIME_CONFIG sets SBMC_OBJDUMP and SBMC_RUNTIME_DIRS (the folders of the runtime DLLs of
# the toolchain). Every DLL imported by the executable (objdump -p, lines "DLL Name: <name>")
# that exists in one of those folders is copied next to it when different, and its own imports
# are followed the same way; an imported DLL found in none of the folders belongs to the system
# and is not copied. The last line names the copied DLLs.
cmake_minimum_required(VERSION 3.28)

foreach(_sbmc_required IN ITEMS SBMC_EXE SBMC_RUNTIME_CONFIG)
  if(NOT DEFINED ${_sbmc_required} OR "${${_sbmc_required}}" STREQUAL "")
    message(FATAL_ERROR "sbmc_copy_runtime.cmake: ${_sbmc_required} is not set")
  endif()
endforeach()
include("${SBMC_RUNTIME_CONFIG}")

get_filename_component(_sbmc_dest "${SBMC_EXE}" DIRECTORY)
get_filename_component(_sbmc_exe_name "${SBMC_EXE}" NAME)
set(_sbmc_queue "${SBMC_EXE}")
set(_sbmc_seen "")
set(_sbmc_copied "")
while(_sbmc_queue)
  list(POP_FRONT _sbmc_queue _sbmc_file)
  execute_process(
    COMMAND "${SBMC_OBJDUMP}" -p "${_sbmc_file}"
    OUTPUT_VARIABLE _sbmc_out
    ERROR_VARIABLE _sbmc_err
    RESULT_VARIABLE _sbmc_status)
  if(NOT _sbmc_status EQUAL 0)
    message(FATAL_ERROR "objdump -p failed on ${_sbmc_file} (${_sbmc_status}): ${_sbmc_err}")
  endif()
  string(REGEX MATCHALL "DLL Name: [^\r\n]+" _sbmc_lines "${_sbmc_out}")
  foreach(_sbmc_line IN LISTS _sbmc_lines)
    string(REGEX REPLACE "^DLL Name: " "" _sbmc_name "${_sbmc_line}")
    string(STRIP "${_sbmc_name}" _sbmc_name)
    string(TOLOWER "${_sbmc_name}" _sbmc_key)
    if(_sbmc_key IN_LIST _sbmc_seen)
      continue()
    endif()
    list(APPEND _sbmc_seen "${_sbmc_key}")
    foreach(_sbmc_dir IN LISTS SBMC_RUNTIME_DIRS)
      if(EXISTS "${_sbmc_dir}/${_sbmc_name}")
        file(COPY_FILE "${_sbmc_dir}/${_sbmc_name}" "${_sbmc_dest}/${_sbmc_name}"
             ONLY_IF_DIFFERENT)
        list(APPEND _sbmc_copied "${_sbmc_name}")
        list(APPEND _sbmc_queue "${_sbmc_dir}/${_sbmc_name}")
        break()
      endif()
    endforeach()
  endforeach()
endwhile()

list(SORT _sbmc_copied CASE INSENSITIVE)
list(JOIN _sbmc_copied " " _sbmc_text)
message(STATUS "Runtime DLLs next to ${_sbmc_exe_name}: ${_sbmc_text}")
