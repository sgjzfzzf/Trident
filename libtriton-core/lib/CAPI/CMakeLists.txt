add_mlir_public_c_api_library(
  LibTritonCoreCAPI
  Dialects.cc
  Registration.cc
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/libtriton-core-c/
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  LibTritonUtilsRegistration
  AOTInductorDialect
  MLIRConvertToLLVMPass
  MLIRIR
  MLIRSupport
  TorchMLIRCAPI
  TorchMLIRTorchConversionDialect)

install(
  TARGETS LibTritonCoreCAPI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
