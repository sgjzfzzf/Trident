add_mlir_public_c_api_library(
  LibTritonCoreCAPI
  Dialects.cc
  Registration.cc
  DLPackTypes.cc
  TVMFFITypes.cc
  DLPackOps.cc
  TVMFFIOps.cc
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/libtriton-core-c/
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  MLIRConvertToLLVMPass
  DLPackDialect
  DLPackToLLVM
  EmitTVMFFIInterface
  MemRefOriginAnalysis
  MLIRIR
  MLIRSupport
  TorchMLIRCAPI
  TorchMLIRTorchConversionDialect
  TorchExtBufferize
  TorchExtToLLVM
  TorchExtNormalizeOperands
  TorchExtDialect
  TVMFFIDialect
  TVMFFIToLLVM)

install(
  TARGETS LibTritonCoreCAPI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
