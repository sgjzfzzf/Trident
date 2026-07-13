# CAPI is now merged into the TorchToLLVM library. It no longer depends on
# TORCH_LIBRARIES (c10::Dispatcher) — the dispatch goes through
# AtenGen-generated TVM FFI global function wrappers instead. CAPI.cc uses
# Torch/C10 headers that require RTTI.
set_source_files_properties(CAPI.cc PROPERTIES COMPILE_FLAGS "-frtti")

add_mlir_conversion_library(
  TorchToLLVM
  TorchToLLVM.cc
  Constant.cc
  Literal.cc
  Aten.cc
  CAPI.cc
  Prim.cc
  FuncBackendTypeConversion.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  TorchExtTransforms
  MLIRLLVMDialect
  MLIRPass
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  PRIVATE
  ${TORCH_LIBRARIES})
