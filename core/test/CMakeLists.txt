if(NOT LLVM_EXTERNAL_LIT OR NOT EXISTS "${LLVM_EXTERNAL_LIT}")
  find_program(TRIDENT_EXTERNAL_LIT NAMES llvm-lit lit REQUIRED)
  set(LLVM_EXTERNAL_LIT "${TRIDENT_EXTERNAL_LIT}")
endif()
set(LLVM_EXTERNAL_LIT
    "${LLVM_EXTERNAL_LIT}"
    CACHE STRING "Command used to spawn lit")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(TRIDENT_CORE_TEST_DEPENDS FileCheck trident-core-opt)

add_lit_testsuite(
  check-trident-core "Running the Trident core regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TRIDENT_CORE_TEST_DEPENDS})

set_target_properties(check-trident-core PROPERTIES FOLDER "Tests")

add_lit_testsuites(TRIDENT_CORE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${TRIDENT_CORE_TEST_DEPENDS})
