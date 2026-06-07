//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "libtriton-core/Runtime/Runtime.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"

namespace libtriton::tvm_ffi::capi {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_create_tensor_from_blob,
                                           AOTITorchCreateTensorFromBlob)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(
    mLibTritonTVMFFIDeviceToTorchDeviceType, TVMFFIDeviceToTorchDeviceType)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTVMFFIToTorchType,
                                           TVMFFIToTorchType)

} // namespace libtriton::tvm_ffi::capi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_
