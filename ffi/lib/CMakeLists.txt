# ===----------------------------------------------------------------------===//
#
# Part of the Trident project, under the MIT License. SPDX-License-Identifier:
# MIT
#
# ===----------------------------------------------------------------------===//

add_library(TridentFFI SHARED Exception.cc)
target_include_directories(TridentFFI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  TridentFFI
  PUBLIC tvm_ffi::header
  PRIVATE tvm_ffi::shared)

# --------------------------------------------------------------------------- #
# tvm-ffi-stubgen: generate `_ffi_api.py` type stubs from libTridentFFI.so
# --------------------------------------------------------------------------- #
set(STUBGEN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_ffi_api.py")

add_custom_command(
  OUTPUT "${STUBGEN_OUTPUT}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/python/_ffi_api.py" "${STUBGEN_OUTPUT}"
  COMMAND ${Python3_EXECUTABLE} -m tvm_ffi.stub.cli
          "${CMAKE_CURRENT_BINARY_DIR}" --dlls "$<TARGET_FILE:TridentFFI>"
  DEPENDS TridentFFI "${CMAKE_SOURCE_DIR}/python/_ffi_api.py"
  COMMENT "Generating tvm-ffi stubs from libTridentFFI.so"
  VERBATIM)

add_custom_target(TridentFFIStubs DEPENDS "${STUBGEN_OUTPUT}")

install(FILES "${STUBGEN_OUTPUT}" "${CMAKE_SOURCE_DIR}/python/__init__.py"
        DESTINATION ffi)

install(
  TARGETS TridentFFI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
