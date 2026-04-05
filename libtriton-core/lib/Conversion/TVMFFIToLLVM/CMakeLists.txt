add_mlir_conversion_library(TVMFFIToLLVM
	TVMFFIToLLVM.cc

	LINK_LIBS PUBLIC
	MLIRPass
	TVMFFIDialect
)
