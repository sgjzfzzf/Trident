add_executable(e2e-dlpack-runner-asan
  e2e-dlpack-runner.c
)

target_compile_options(e2e-dlpack-runner-asan PRIVATE
  -fsanitize=address
  -fno-omit-frame-pointer
  -g
)

target_link_options(e2e-dlpack-runner-asan PRIVATE
  -fsanitize=address
)

target_link_libraries(e2e-dlpack-runner-asan PRIVATE
  tvm_ffi::shared
)

set_target_properties(e2e-dlpack-runner-asan PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

set(INTEGRATION_TEST_TARGETS e2e-dlpack-runner-asan)

# Make it available to lit tests
set(INTEGRATION_TEST_TARGETS ${INTEGRATION_TEST_TARGETS} PARENT_SCOPE)
