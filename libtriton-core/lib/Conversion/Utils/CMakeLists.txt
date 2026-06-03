add_mlir_conversion_library(
  ConversionUtils
  IConstantUtils.cc
  LINK_LIBS
  PUBLIC
  MLIRLLVMDialect
  MLIRTransforms)
