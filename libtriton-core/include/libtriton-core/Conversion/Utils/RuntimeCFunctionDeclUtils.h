#ifndef LIBTRITON_CORE_CONVERSION_UTILS_RUNTIMECFUNCTIONDECLUTILS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_RUNTIMECFUNCTIONDECLUTILS_H_

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "libtriton-core/Runtime/Runtime.h"

namespace libtriton::conversion::utils::runtime {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(
    __libtriton_dlpack_default_managed_tensor_deleter,
    DefaultManagedTensorDeleter)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(__libtriton_get_current_device,
                                           GetCurrentDevice)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(__libtriton_get_current_stream,
                                           GetCurrentStream)

} // namespace libtriton::conversion::utils::runtime

#endif // LIBTRITON_CORE_CONVERSION_UTILS_RUNTIMECFUNCTIONDECLUTILS_H_
