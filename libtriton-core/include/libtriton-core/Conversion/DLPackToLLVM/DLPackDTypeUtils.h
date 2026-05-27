#ifndef LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKDTYPEUTILS_H_
#define LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKDTYPEUTILS_H_

#include <optional>

#include "dlpack/dlpack.h"
#include "mlir/IR/Types.h"

namespace libtriton::conversion::utils {

std::optional<DLDataType> getDLPackDTypeFromMLIRType(mlir::Type type);

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKDTYPEUTILS_H_
