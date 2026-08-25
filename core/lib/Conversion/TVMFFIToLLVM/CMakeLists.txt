add_mlir_conversion_library(
  TVMFFIToLLVM
  TVMFFIToLLVM.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  ConversionUtils
  MLIRControlFlowTransforms
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  TVMFFIDialect)

target_link_libraries(TVMFFIToLLVM PUBLIC ${TORCH_LIBRARIES})
