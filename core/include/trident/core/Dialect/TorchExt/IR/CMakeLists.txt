add_mlir_dialect(TorchExt torchext)

set(LLVM_TARGET_DEFINITIONS TorchExt.td)
mlir_tablegen(TorchExtTypes.h.inc -gen-typedef-decls -typedefs-dialect=torchext)
mlir_tablegen(TorchExtTypes.cpp.inc -gen-typedef-defs
              -typedefs-dialect=torchext)
add_public_tablegen_target(MLIRTorchExtTypeDefsIncGen)
add_dependencies(MLIRTorchExtIncGen MLIRTorchExtTypeDefsIncGen)
