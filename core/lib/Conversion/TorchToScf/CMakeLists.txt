set(LLVM_TARGET_DEFINITIONS TorchToScf.td)
mlir_tablegen(TorchToScfPatterns.inc -gen-rewriters)
add_public_tablegen_target(TorchToScfPatternsIncGen)

add_mlir_conversion_library(
  TorchToScf
  TorchToScf.cc
  DEPENDS
  MLIRConversionPassIncGen
  TorchToScfPatternsIncGen
  LINK_LIBS
  PUBLIC
  MLIRSCFDialect
  MLIRTransforms
  TorchExtDialect
  TorchMLIRTorchDialect)

target_include_directories(TorchToScf PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
