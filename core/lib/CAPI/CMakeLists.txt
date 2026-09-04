add_mlir_public_c_api_library(
  TridentCoreCAPI
  Dialects.cc
  Registration.cc
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRIR
  MLIRSupport
  TorchExtDialect
  TorchMLIRCAPI
  TorchMLIRTorchConversionDialect
  TorchToTVMFFI
  TridentUtilsRegistration
  TVMFFIDialect)

install(
  TARGETS TridentCoreCAPI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
