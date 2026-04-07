add_mlir_conversion_library(ConvertFuncSignatureToDLPack
  ConvertFuncSignatureToDLPack.cc

  LINK_LIBS PUBLIC
  DLPackDialect
  MLIRFuncDialect
  MLIRPass
)
