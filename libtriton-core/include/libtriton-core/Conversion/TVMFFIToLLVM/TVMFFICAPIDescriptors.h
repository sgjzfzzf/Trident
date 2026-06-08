//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_

#include "dlpack/dlpack.h"
#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "libtriton-core/Runtime/Runtime.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"
#include "tvm/ffi/c_api.h"

namespace libtriton::tvm_ffi::capi {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_create_tensor_from_blob,
                                           AOTITorchCreateTensorFromBlob)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(
    mLibTritonTVMFFIDeviceToTorchDeviceType, TVMFFIDeviceToTorchDeviceType)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTVMFFIToTorchType,
                                           TVMFFIToTorchType)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonDLManagedTensorDeleter,
                                           DLManagedTensorDeleter)

// AOTI tensor property accessors.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_data_ptr,
                                           AOTITorchGetDataPtr)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_dim, AOTITorchGetDim)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_sizes,
                                           AOTITorchGetSizes)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_strides,
                                           AOTITorchGetStrides)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_dtype,
                                           AOTITorchGetDtype)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_device_type,
                                           AOTITorchGetDeviceType)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_device_index,
                                           AOTITorchGetDeviceIndex)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_storage_offset,
                                           AOTITorchGetStorageOffset)

// TVM FFI tensor conversion.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(TVMFFITensorFromDLPack,
                                           TVMFFITensorFromDLPack)

// Reverse Torch→DLPack dtype/device mapping.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTorchToTVMFFIDtype,
                                           TorchToTVMFFIDtype)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTorchToTVMFFIDevice,
                                           TorchToTVMFFIDevice)

// Runtime tensor pack/unpack helpers.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonPackTensorToTVMFFIAny,
                                           PackTensorToTVMFFIAny)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonUnpackTVMFFIAnyToTensor,
                                           UnpackTVMFFIAnyToTensor)

// Standard C library.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(malloc, Malloc)

// TVM FFI error reporting.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(TVMFFIErrorSetRaisedFromCStr)

} // namespace libtriton::tvm_ffi::capi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFICAPIDESCRIPTORS_H_
