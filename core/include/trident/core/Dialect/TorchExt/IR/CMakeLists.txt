add_mlir_dialect(TorchExt torchext)

set(LLVM_TARGET_DEFINITIONS TorchExt.td)
mlir_tablegen(TorchExtAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=torchext)
mlir_tablegen(TorchExtAttrs.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=torchext)
add_public_tablegen_target(MLIRTorchExtAttrDefsIncGen)
mlir_tablegen(TorchExtTypes.h.inc -gen-typedef-decls -typedefs-dialect=torchext)
mlir_tablegen(TorchExtTypes.cpp.inc -gen-typedef-defs
              -typedefs-dialect=torchext)
add_public_tablegen_target(MLIRTorchExtTypeDefsIncGen)
add_dependencies(MLIRTorchExtIncGen MLIRTorchExtAttrDefsIncGen)
add_dependencies(MLIRTorchExtIncGen MLIRTorchExtTypeDefsIncGen)
