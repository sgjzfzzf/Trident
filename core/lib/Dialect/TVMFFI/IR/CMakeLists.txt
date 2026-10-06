add_mlir_dialect_library(
  TVMFFIDialect
  TVMFFIDialect.cc
  TVMFFIInterfaces.cc
  TVMFFIOps.cc
  TVMFFISpecialization.cc
  TVMFFITypes.cc
  DEPENDS
  MLIRTVMFFIAttrDefsIncGen
  MLIRTVMFFIAttrInterfacesIncGen
  MLIRTVMFFIIncGen
  MLIRTVMFFIInterfacesIncGen
  MLIRTVMFFITypeInterfacesIncGen
  MLIRTVMFFITypeDefsIncGen
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  tvm_ffi::shared)

target_include_directories(TVMFFIDialect
                           PRIVATE ${CMAKE_SOURCE_DIR}/../ffi/include)
