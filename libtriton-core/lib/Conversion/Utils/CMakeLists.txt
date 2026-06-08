add_mlir_conversion_library(
  ConversionUtils
  GlobalString.cc
  IConstantUtils.cc
  LINK_LIBS
  PUBLIC
  MLIRLLVMDialect
  MLIRTransforms)
