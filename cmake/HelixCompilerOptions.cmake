include_guard(GLOBAL)

function(helix_configure_target target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    if(HELIX_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
        $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>)
    if(HELIX_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
    endif()
    if(HELIX_ENABLE_CUDA_LINEINFO)
      target_compile_options(
        ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()
  endif()

  if(HELIX_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND NOT WIN32)
      target_compile_options(
        ${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
      target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
    else()
      message(FATAL_ERROR
        "HELIX_ENABLE_SANITIZERS currently requires Clang or GCC on a non-Windows host")
    endif()
  endif()

  if(HELIX_ENABLE_CLANG_TIDY)
    find_program(HELIX_CLANG_TIDY_EXECUTABLE NAMES clang-tidy)
    if(NOT HELIX_CLANG_TIDY_EXECUTABLE AND APPLE)
      execute_process(
        COMMAND xcrun --find clang-tidy
        RESULT_VARIABLE _helix_tidy_result
        OUTPUT_VARIABLE _helix_tidy_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(_helix_tidy_result EQUAL 0)
        set(HELIX_CLANG_TIDY_EXECUTABLE "${_helix_tidy_path}")
      endif()
    endif()
    if(NOT HELIX_CLANG_TIDY_EXECUTABLE)
      message(FATAL_ERROR
        "HELIX_ENABLE_CLANG_TIDY=ON, but clang-tidy was not found")
    endif()
    set_property(
      TARGET ${target_name}
      PROPERTY CXX_CLANG_TIDY
        "${HELIX_CLANG_TIDY_EXECUTABLE};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy")
  endif()
endfunction()

function(helix_add_format_targets)
  find_program(HELIX_CLANG_FORMAT_EXECUTABLE NAMES clang-format)
  if(NOT HELIX_CLANG_FORMAT_EXECUTABLE AND APPLE)
    execute_process(
      COMMAND xcrun --find clang-format
      RESULT_VARIABLE _helix_format_result
      OUTPUT_VARIABLE _helix_format_path
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(_helix_format_result EQUAL 0)
      set(HELIX_CLANG_FORMAT_EXECUTABLE "${_helix_format_path}")
    endif()
  endif()

  file(
    GLOB_RECURSE HELIX_FORMAT_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/tests/*.cpp"
    "${PROJECT_SOURCE_DIR}/benchmarks/*.cpp"
    "${PROJECT_SOURCE_DIR}/examples/*.cpp"
    "${PROJECT_SOURCE_DIR}/tools/*.cpp")

  if(HELIX_CLANG_FORMAT_EXECUTABLE)
    add_custom_target(
      format
      COMMAND "${HELIX_CLANG_FORMAT_EXECUTABLE}" -i ${HELIX_FORMAT_FILES}
      COMMENT "Formatting Helix C++ and CUDA sources"
      VERBATIM)
    add_custom_target(
      format-check
      COMMAND "${HELIX_CLANG_FORMAT_EXECUTABLE}" --dry-run --Werror ${HELIX_FORMAT_FILES}
      COMMENT "Checking Helix C++ and CUDA formatting"
      VERBATIM)
  else()
    add_custom_target(
      format
      COMMAND "${CMAKE_COMMAND}" -E echo "clang-format is required for target 'format'"
      COMMAND "${CMAKE_COMMAND}" -E false
      VERBATIM)
    add_custom_target(
      format-check
      COMMAND "${CMAKE_COMMAND}" -E echo "clang-format is required for target 'format-check'"
      COMMAND "${CMAKE_COMMAND}" -E false
      VERBATIM)
  endif()
endfunction()
