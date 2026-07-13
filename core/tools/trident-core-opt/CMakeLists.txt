add_mlir_tool(trident-core-opt trident-core-opt.cc)

set_target_properties(trident-core-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${PROJECT_BINARY_DIR}/bin")

target_link_libraries(
  trident-core-opt
  PRIVATE TorchExtDialect TridentUtilsRegistration MLIRMlirOptMain
          TVMFFIDialect CUDA::cudart CUDA::cupti)

mlir_check_all_link_libraries(trident-core-opt)
