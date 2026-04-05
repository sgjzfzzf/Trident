#ifndef LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_

#include <cstdint>
#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace libtriton::dlpack {

void populateDLPackToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter);

std::unique_ptr<mlir::Pass> createConvertDLPackToLLVMPass();
void registerConvertDLPackToLLVMPass();
void registerDLPackToLLVMPasses();

} // namespace libtriton::dlpack

#endif // LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
