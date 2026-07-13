//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "CAPI.h"
#include "ATen/DLConvertor.h"

DLDevice torchDeviceToDLDevice(const char *device) {
  // Use PyTorch's stable C API to parse the device string (e.g. "cuda:0",
  // "cpu").
  return at::torchDeviceToDLDevice(at::Device(device));
}
