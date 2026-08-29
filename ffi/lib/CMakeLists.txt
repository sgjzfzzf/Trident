# ===----------------------------------------------------------------------===//
#
# Part of the Trident project, under the MIT License. SPDX-License-Identifier:
# MIT
#
# ===----------------------------------------------------------------------===//

add_library(TridentFFI SHARED Exception.cc Runtime.cc)
target_include_directories(TridentFFI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  TridentFFI
  PUBLIC ${TORCH_LIBRARIES} tvm_ffi::header
  PRIVATE tvm_ffi::shared)

# --------------------------------------------------------------------------- #
# ATen operator wrappers exposed through the TVM FFI global function table.
# --------------------------------------------------------------------------- #
set(ATENGEN_CC "${CMAKE_CURRENT_BINARY_DIR}/aten.gen.cc")
set(ATENGEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/aten/atengen.py")
set(ATENGEN_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/aten/aten.cc.j2")

add_custom_command(
  OUTPUT "${ATENGEN_CC}"
  COMMAND ${Python3_EXECUTABLE} "${ATENGEN_PY}" --out "${ATENGEN_CC}"
  COMMAND clang-format -i "${ATENGEN_CC}"
  DEPENDS "${ATENGEN_PY}" "${ATENGEN_TEMPLATE}"
  COMMENT "Generating ${ATENGEN_CC}"
  VERBATIM)

add_library(TridentAtenFFI SHARED "${ATENGEN_CC}")
set_source_files_properties("${ATENGEN_CC}" PROPERTIES SKIP_LINTING TRUE)
target_include_directories(TridentAtenFFI
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/aten")
target_link_libraries(
  TridentAtenFFI
  PUBLIC ${TORCH_LIBRARIES} tvm_ffi::header
  PRIVATE tvm_ffi::shared)

install(
  TARGETS TridentAtenFFI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

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

add_custom_target(TridentFFIStubs ALL DEPENDS "${STUBGEN_OUTPUT}")

install(FILES "${STUBGEN_OUTPUT}" "${CMAKE_SOURCE_DIR}/python/__init__.py"
        DESTINATION ffi)

install(
  TARGETS TridentFFI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
