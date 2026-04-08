execute_process(
  COMMAND which lit
  OUTPUT_VARIABLE LLVM_EXTERNAL_LIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(LLVM_EXTERNAL_LIT "${LLVM_EXTERNAL_LIT}" CACHE STRING "Command used to spawn lit" FORCE)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Build integration test tools.
add_subdirectory(Integration)
add_subdirectory(CAPI)

set(LIBTRITON_CORE_TEST_DEPENDS
  FileCheck
  ${CAPI_TEST_TARGETS}
  libtriton-core-opt
  ${INTEGRATION_TEST_TARGETS}
)

add_lit_testsuite(check-libtriton-core "Running the TVM FFI bindings regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LIBTRITON_CORE_TEST_DEPENDS}
)

set_target_properties(check-libtriton-core PROPERTIES FOLDER "Tests")

add_lit_testsuites(LIBTRITON_CORE ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${LIBTRITON_CORE_TEST_DEPENDS}
)
