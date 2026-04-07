add_mlir_tool(libtriton-core-opt
  libtriton-core-opt.cc
)

set_target_properties(libtriton-core-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

target_link_libraries(libtriton-core-opt PRIVATE
  ConvertFuncSignatureToDLPack
  DLPackDialect
  DLPackToLLVM
  MLIRArithDialect
  MLIRConvertToLLVMPass
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMlirOptMain
  MLIRSCFDialect
  MLIRTransforms
  TVMFFIToLLVM
  TVMFFIDialect
)

mlir_check_all_link_libraries(libtriton-core-opt)
