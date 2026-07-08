//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_CAPI_H_
#define TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_CAPI_H_

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif

DLDevice torchDeviceToDLDevice(const char *device);

#ifdef __cplusplus
}
#endif

#endif // TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_CAPI_H_
