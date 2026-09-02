add_mlir_pdll_library(
  GeneralizeAtenOpsPDLLPatternsIncGen
  GeneralizeAtenOps.pdll
  GeneralizeAtenOpsPDLLPatterns.h.inc
  EXTRA_INCLUDES
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${torch-mlir_SOURCE_DIR}/include
  ${torch-mlir_BINARY_DIR}/include)

add_mlir_library(
  GeneralizeAtenOps
  GeneralizeAtenOps.cc
  DEPENDS
  MLIRTorchTransformsPassIncGen
  GeneralizeAtenOpsPDLLPatternsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRArithDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect)

target_include_directories(GeneralizeAtenOps
                           PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
