add_mlir_conversion_library(TVMFFIToLLVM
	TVMFFIToLLVM.cc

	DEPENDS
	MLIRConversionPassIncGen

	LINK_LIBS PUBLIC
	MemRefOriginAnalysis
	MLIRPass
	TVMFFIDialect
	tvm_ffi::header
)
