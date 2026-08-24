add_executable(helix_model_pack model_pack.cpp)
target_link_libraries(helix_model_pack PRIVATE helix::model)
helix_configure_target(helix_model_pack)

add_executable(helix_model_inspect model_inspect.cpp)
target_link_libraries(helix_model_inspect PRIVATE helix::model)
helix_configure_target(helix_model_inspect)

install(
  TARGETS helix_model_pack helix_model_inspect
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(HELIX_BUILD_TESTS)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${HELIX_CONVERTER_PYTHONPATH}"
        "${Python3_EXECUTABLE}" -c "import numpy"
      RESULT_VARIABLE HELIX_CONVERTER_NUMPY_RESULT
      OUTPUT_QUIET
      ERROR_QUIET)
  endif()

  if(Python3_Interpreter_FOUND AND HELIX_CONVERTER_NUMPY_RESULT EQUAL 0)
    add_test(
      NAME model_converter.integration
      COMMAND
        "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${HELIX_CONVERTER_PYTHONPATH}"
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_convert_hf_llama.py"
        --converter "${CMAKE_CURRENT_SOURCE_DIR}/convert_hf_llama.py"
        --validator "${CMAKE_CURRENT_SOURCE_DIR}/validate_hf_llama.py"
        --packer "$<TARGET_FILE:helix_model_pack>"
        --inspector "$<TARGET_FILE:helix_model_inspect>")
    set_tests_properties(
      model_converter.integration
      PROPERTIES LABELS converter TIMEOUT 180)
    message(STATUS "  Converter tests:    ON (${Python3_EXECUTABLE})")
  else()
    message(STATUS "  Converter tests:    OFF (Python 3 with NumPy not found)")
  endif()
endif()
