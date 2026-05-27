#include "libtriton-core/Conversion/DLPackToLLVM/DLPackDTypeUtils.h"

#include <cstdint>
#include <optional>

#include "mlir/IR/BuiltinTypes.h"

namespace libtriton::conversion::utils {

std::optional<DLDataType> getDLPackDTypeFromMLIRType(mlir::Type type) {
  if (type.isF16()) {
    return DLDataType{static_cast<std::uint8_t>(kDLFloat), 16, 1};
  } else if (type.isF32()) {
    return DLDataType{static_cast<std::uint8_t>(kDLFloat), 32, 1};
  } else if (type.isF64()) {
    return DLDataType{static_cast<std::uint8_t>(kDLFloat), 64, 1};
  } else if (type.isBF16()) {
    return DLDataType{static_cast<std::uint8_t>(kDLBfloat), 16, 1};
  } else if (mlir::IntegerType integerType =
                 mlir::dyn_cast<mlir::IntegerType>(type)) {
    const std::uint8_t code = integerType.isUnsigned()
                                  ? static_cast<std::uint8_t>(kDLUInt)
                                  : static_cast<std::uint8_t>(kDLInt);
    return DLDataType{code, static_cast<std::uint8_t>(integerType.getWidth()),
                      1};
  } else {
    return std::nullopt;
  }
}

} // namespace libtriton::conversion::utils
