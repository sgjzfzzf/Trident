#ifndef TVM_FFI_BINDINGS_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
#define TVM_FFI_BINDINGS_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_

#include <cstdint>
#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

namespace libtriton::dlpack {

mlir::LLVM::LLVMStructType getDLContextLLVMType(mlir::MLIRContext *context);
mlir::LLVM::LLVMStructType getDLDataTypeLLVMType(mlir::MLIRContext *context);
mlir::LLVM::LLVMStructType getDLTensorLLVMType(mlir::MLIRContext *context,
                                               std::uint32_t sizeTWidth);

void populateDLPackToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter);

std::unique_ptr<mlir::Pass> createConvertDLPackToLLVMPass();
void registerConvertDLPackToLLVMPass();
void registerDLPackToLLVMPasses();

} // namespace libtriton::dlpack

#endif // TVM_FFI_BINDINGS_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
