add_mlir_public_c_api_library(LibTritonCoreCAPI
  Dialects.cpp
  Registration.cpp
  DLPackTypes.cpp
  TVMFFITypes.cpp
  DLPackOps.cpp
  TVMFFIOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/libtriton-core-c/

  ENABLE_AGGREGATION

  LINK_LIBS PUBLIC
  DLPackDialect
  DLPackToLLVM
  MemRefOriginAnalysis
  MLIRIR
  MLIRSupport
  TorchMLIRCAPI
  TVMFFIDialect
  TVMFFIToLLVM
)

install(TARGETS LibTritonCoreCAPI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
