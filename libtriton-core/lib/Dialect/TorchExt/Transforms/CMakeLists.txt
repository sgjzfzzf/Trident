add_mlir_library(
  TorchExtBufferize
  BufferizableOpInterfaceImpl.cc
  LINK_LIBS
  PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  TorchExtDialect)
