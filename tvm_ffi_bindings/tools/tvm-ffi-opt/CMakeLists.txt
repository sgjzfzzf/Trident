add_mlir_tool(tvm-ffi-opt
  tvm-ffi-opt.cc
)

set_target_properties(tvm-ffi-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

target_link_libraries(tvm-ffi-opt PRIVATE
  DLPackDialect
  DLPackToLLVM
  MLIRConvertToLLVMPass
  MLIRFuncDialect
  MLIRMlirOptMain
  TVMFFIToLLVM
  TVMFFIDialect
)

mlir_check_all_link_libraries(tvm-ffi-opt)
