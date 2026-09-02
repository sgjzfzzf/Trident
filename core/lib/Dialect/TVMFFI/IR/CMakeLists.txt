add_mlir_dialect_library(
  TVMFFIDialect
  TVMFFIDialect.cc
  TVMFFIInterfaces.cc
  TVMFFIOps.cc
  TVMFFITypes.cc
  DEPENDS
  MLIRTVMFFIIncGen
  MLIRTVMFFIInterfacesIncGen
  MLIRTVMFFITypeInterfacesIncGen
  MLIRTVMFFITypeDefsIncGen
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  tvm_ffi::shared)

target_include_directories(TVMFFIDialect
                           PRIVATE ${CMAKE_SOURCE_DIR}/../ffi/include)
