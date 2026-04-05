#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace libtriton::tvm_ffi {

std::unique_ptr<mlir::Pass> createConvertTVMFFIToLLVMPass();
void registerConvertTVMFFIToLLVMPass();
void registerTVMFFIToLLVMPasses();

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
