add_mlir_dialect(TVMFFI tvm_ffi)

set(LLVM_TARGET_DEFINITIONS TVMFFI.td)
mlir_tablegen(TVMFFIAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=tvm_ffi)
mlir_tablegen(TVMFFIAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=tvm_ffi)
mlir_tablegen(TVMFFITypes.h.inc -gen-typedef-decls -typedefs-dialect=tvm_ffi)
mlir_tablegen(TVMFFITypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tvm_ffi)
add_public_tablegen_target(MLIRTVMFFIAttrDefsIncGen)
add_public_tablegen_target(MLIRTVMFFITypeDefsIncGen)
add_dependencies(MLIRTVMFFIIncGen MLIRTVMFFIAttrDefsIncGen
                 MLIRTVMFFITypeDefsIncGen)
