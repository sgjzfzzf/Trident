add_mlir_dialect(DLPack dlpack)

set(LLVM_TARGET_DEFINITIONS DLPack.td)
mlir_tablegen(DLPackTypes.h.inc -gen-typedef-decls -typedefs-dialect=dlpack)
mlir_tablegen(DLPackTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=dlpack)
add_public_tablegen_target(MLIRDLPackTypeDefsIncGen)
add_dependencies(MLIRDLPackIncGen MLIRDLPackTypeDefsIncGen)
