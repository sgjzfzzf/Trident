set(LLVM_TARGET_DEFINITIONS TorchInterfaces.td)
mlir_tablegen(TorchTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TorchTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRTridentTorchTypeInterfacesIncGen)
