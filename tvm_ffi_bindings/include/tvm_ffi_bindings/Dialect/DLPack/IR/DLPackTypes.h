#ifndef TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_H_
#define TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_H_

#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.h.inc"

namespace libtriton::dlpack {

enum class DLDeviceType : std::int32_t {
  kCPU = 1,
  kCPUPinned = 3,
  kGPU = 2,
  kOpenCL = 4,
};

enum class DLDataTypeCode : std::uint8_t {
  kFloat = 2,
  kInt = 0,
  kUInt = 1,
};

} // namespace libtriton::dlpack

#endif // TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_H_