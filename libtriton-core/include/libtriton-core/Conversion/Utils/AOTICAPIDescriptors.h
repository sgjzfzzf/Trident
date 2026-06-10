//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_AOTICAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_AOTICAPIDESCRIPTORS_H_

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"

// AOTI (PyTorch Inductor) C API function descriptors.
namespace libtriton::conversion::utils {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(aoti_torch_call_dispatcher)

// AOTI tensor property accessors.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_create_tensor_from_blob,
                                           AOTITorchCreateTensorFromBlob)
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

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_AOTICAPIDESCRIPTORS_H_
