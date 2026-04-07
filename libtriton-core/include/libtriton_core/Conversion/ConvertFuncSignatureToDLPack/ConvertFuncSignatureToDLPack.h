#ifndef LIBTRITON_CORE_CONVERSION_CONVERTFUNCSIGNATURETODLPACK_CONVERTFUNCSIGNATURETODLPACK_H_
#define LIBTRITON_CORE_CONVERSION_CONVERTFUNCSIGNATURETODLPACK_CONVERTFUNCSIGNATURETODLPACK_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace libtriton::dlpack {

std::unique_ptr<mlir::Pass> createConvertFuncSignatureToDLPackPass();
void registerConvertFuncSignatureToDLPackPass();

} // namespace libtriton::dlpack

#endif // LIBTRITON_CORE_CONVERSION_CONVERTFUNCSIGNATURETODLPACK_CONVERTFUNCSIGNATURETODLPACK_H_
