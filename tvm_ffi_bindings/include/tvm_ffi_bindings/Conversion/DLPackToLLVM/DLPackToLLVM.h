#ifndef TVM_FFI_BINDINGS_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
#define TVM_FFI_BINDINGS_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace libtriton::dlpack {

std::unique_ptr<mlir::Pass> createConvertDLPackToLLVMPass();
void registerConvertDLPackToLLVMPass();
void registerDLPackToLLVMPasses();

} // namespace libtriton::dlpack

#endif // TVM_FFI_BINDINGS_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
