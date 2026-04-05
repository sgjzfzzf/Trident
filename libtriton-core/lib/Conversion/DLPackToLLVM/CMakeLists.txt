add_mlir_conversion_library(DLPackToLLVM
        DLPackToLLVM.cc

        LINK_LIBS PUBLIC
        DLPackDialect
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRLLVMDialect
        MLIRLLVMCommonConversion
        MLIRPass
        MLIRTransforms
)
