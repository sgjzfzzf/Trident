add_executable(e2e-dlpack-runner
  e2e-dlpack-runner.c
)

# Link tvm-ffi runtime C API.
target_link_libraries(e2e-dlpack-runner PRIVATE
  tvm_ffi::shared
)

set_target_properties(e2e-dlpack-runner PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

# Make it available to lit tests
set(INTEGRATION_TEST_TARGETS e2e-dlpack-runner PARENT_SCOPE)
