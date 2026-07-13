//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_AOTICAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_AOTICAPIDESCRIPTORS_H_

#include "torch/csrc/inductor/aoti_torch/c/shim.h"
#include "torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h"
#include "trident/core/Conversion/Utils/CFunctionDeclUtils.h"

// AOTI (PyTorch Inductor) C API function descriptors.
namespace trident::conversion::utils {

TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_aten_full,
                                         AOTITorchAtenFull)

// AOTI tensor property accessors.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_create_tensor_from_blob,
                                         AOTITorchCreateTensorFromBlob)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_empty_strided,
                                         AOTITorchEmptyStrided)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_copy_, AOTITorchCopy_)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_delete_tensor_object,
                                         AOTITorchDeleteTensorObject)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_data_ptr,
                                         AOTITorchGetDataPtr)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_dim, AOTITorchGetDim)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_sizes,
                                         AOTITorchGetSizes)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_strides,
                                         AOTITorchGetStrides)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_dtype,
                                         AOTITorchGetDtype)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_device_type,
                                         AOTITorchGetDeviceType)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_device_index,
                                         AOTITorchGetDeviceIndex)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_storage_offset,
                                         AOTITorchGetStorageOffset)

// AOTI device/stream helpers.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_current_device_index,
                                         AOTITorchGetCurrentDeviceIndex)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(aoti_torch_get_current_stream,
                                         AOTITorchGetCurrentStream)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_AOTICAPIDESCRIPTORS_H_
