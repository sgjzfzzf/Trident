//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_STABLECAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_STABLECAPIDESCRIPTORS_H_

#include "torch/csrc/stable/c/shim.h"
#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"

// Stable C API function descriptors.
//
// These map 1:1 to functions declared in torch/csrc/stable/c/shim.h, which
// provides ABI-compatible wrappers around PyTorch ops for forward/backward
// compatibility when calling ATen operations through the dispatcher.
namespace trident::conversion::utils {

// StableList construction.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_new_list_reserve_size,
                                         TorchNewListReserveSize)

// StableList introspection.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_size, TorchListSize)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_get_item, TorchListGetItem)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_set_item, TorchListSetItem)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_push_back,
                                         TorchListPushBack)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_delete_list, TorchDeleteList)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_STABLECAPIDESCRIPTORS_H_
