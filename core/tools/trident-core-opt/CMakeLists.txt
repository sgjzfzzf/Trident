add_mlir_tool(trident-core-opt trident-core-opt.cc)

set_target_properties(trident-core-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${PROJECT_BINARY_DIR}/bin")

target_link_libraries(
  trident-core-opt
  PRIVATE CUDA::cudart CUDA::cupti MLIRMlirOptMain TVMFFIDialect
          TridentUtilsRegistration ${TORCH_LIBRARIES})

mlir_check_all_link_libraries(trident-core-opt)
