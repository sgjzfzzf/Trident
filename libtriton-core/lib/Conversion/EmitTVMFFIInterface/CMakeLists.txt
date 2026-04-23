add_mlir_conversion_library(EmitTVMFFIInterface
        EmitTVMFFIInterface.cc

        DEPENDS
        MLIRConversionPassIncGen

        LINK_LIBS PUBLIC
        DLPackDialect
        MemRefOriginAnalysis
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRPass
        TVMFFIDialect
)
