add_mlir_dialect_library(TVMFFIDialect
  TVMFFIDialect.cc
  DEPENDS
  MLIRTVMFFIIncGen
  MLIRDLPackIncGen
  LINK_LIBS PUBLIC
  DLPackDialect
)
