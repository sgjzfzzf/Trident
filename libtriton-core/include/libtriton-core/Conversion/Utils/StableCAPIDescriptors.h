//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_STABLECAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_STABLECAPIDESCRIPTORS_H_

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "torch/csrc/stable/c/shim.h"

// Stable C API function descriptors.
//
// These map 1:1 to functions declared in torch/csrc/stable/c/shim.h, which
// provides ABI-compatible wrappers around PyTorch ops for forward/backward
// compatibility when calling ATen operations through the dispatcher.
namespace libtriton::conversion::utils {

// StableList construction.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_new_list_reserve_size,
                                           TorchNewListReserveSize)

// StableList introspection.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_size, TorchListSize)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_get_item,
                                           TorchListGetItem)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_set_item,
                                           TorchListSetItem)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_list_push_back,
                                           TorchListPushBack)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(torch_delete_list, TorchDeleteList)

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_STABLECAPIDESCRIPTORS_H_
