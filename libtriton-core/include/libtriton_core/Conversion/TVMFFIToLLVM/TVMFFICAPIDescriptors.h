#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_

#include <cstdint>

#include "libtriton_core/Conversion/Utils/CFunctionDeclUtils.h"
#include "tvm/ffi/c_api.h"

namespace libtriton::tvm_ffi::capi {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(TVMFFITensorFromDLPack)

} // namespace libtriton::tvm_ffi::capi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_
