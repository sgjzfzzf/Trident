add_mlir_tool(libtriton-core-opt libtriton-core-opt.cc)

set_target_properties(libtriton-core-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${PROJECT_BINARY_DIR}/bin")

target_link_libraries(
  libtriton-core-opt
  PRIVATE DLPackDialect
          DLPackToLLVM
          EmitTVMFFIInterface
          MemRefOriginAnalysis
          TorchExtToLLVM
          TorchExtAsyncKernelLaunch
          TorchExtNormalizeOperands
          MLIRArithDialect
          MLIRConvertToLLVMPass
          MLIRControlFlowDialect
          MLIRFuncDialect
          MLIRGPUDialect
          MLIRMemRefDialect
          MLIRMlirOptMain
          MLIRSCFDialect
          MLIRTransforms
          TorchMLIRTorchConversionDialect
          TorchExtDialect
          TVMFFIToLLVM
          TVMFFIDialect)

mlir_check_all_link_libraries(libtriton-core-opt)
