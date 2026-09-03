add_mlir_pdll_library(
  TVMFFIToLLVMPDLLPatternsIncGen
  TVMFFIToLLVM.pdll
  TVMFFIToLLVMPDLLPatterns.h.inc
  EXTRA_INCLUDES
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${LLVM_MAIN_SRC_DIR}/include
  ${LLVM_BINARY_DIR}/include)

add_mlir_conversion_library(
  TVMFFIToLLVM
  TVMFFIToLLVM.cc
  DEPENDS
  MLIRConversionPassIncGen
  TVMFFIToLLVMPDLLPatternsIncGen
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

target_include_directories(TVMFFIToLLVM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
