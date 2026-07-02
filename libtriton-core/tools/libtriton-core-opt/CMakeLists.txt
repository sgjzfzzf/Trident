add_mlir_tool(libtriton-core-opt libtriton-core-opt.cc)

set_target_properties(libtriton-core-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${PROJECT_BINARY_DIR}/bin")

target_link_libraries(
  libtriton-core-opt
  PRIVATE TorchExtDialect LibTritonUtilsRegistration MLIRMlirOptMain
          TVMFFIDialect "${Python3_SITELIB}/nvidia/cu13/lib/libcudart.so.13"
          "${Python3_SITELIB}/nvidia/cu13/lib/libcupti.so.13")

mlir_check_all_link_libraries(libtriton-core-opt)
