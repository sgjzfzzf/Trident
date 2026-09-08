add_mlir_pdll_library(
  TorchToTVMFFIPDLLPatternsIncGen
  TorchToTVMFFI.pdll
  TorchToTVMFFIPDLLPatterns.h.inc
  EXTRA_INCLUDES
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${torch-mlir_SOURCE_DIR}/include
  ${torch-mlir_BINARY_DIR}/include)

add_mlir_conversion_library(
  TorchToTVMFFI
  TorchToTVMFFI.cc
  DEPENDS
  MLIRConversionPassIncGen
  MLIRTridentTorchTypeInterfacesIncGen
  TorchToTVMFFIPDLLPatternsIncGen
  LINK_LIBS
  PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  TorchExtDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TridentTorchInterfaces
  TVMFFIDialect)

target_include_directories(TorchToTVMFFI PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
