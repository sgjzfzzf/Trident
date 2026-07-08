//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Runtime/Runtime.h"
#include "dlpack/dlpack.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"
#include <cstdlib>

/// DLPack-compatible deleter callback — frees the AtenTensorHandle stored in
/// self->manager_ctx, then frees the DLManagedTensor allocation.
/// Used only internally by mTridentTensorToTVMFFIObject.
static void mTridentDLManagedTensorDeleter(struct DLManagedTensor *self) {
  if (self) {
    if (self->manager_ctx) {
      aoti_torch_delete_tensor_object(
          static_cast<AtenTensorHandle>(self->manager_ctx));
    }
    std::free(self);
  }
}

/// Check that a call returns 0 (success), or that a condition is false;
/// return -1 on failure.
#define TRIDENT_TVMFFI_CHECK(cond)                                             \
  do {                                                                         \
    if (cond) {                                                                \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// X-Macro: (DLDataTypeCode, bits, aoti_torch_dtype_* function name)
#define TRIDENT_TVMFFI_DTYPE_PAIR(X)                                           \
  X(kDLBfloat, 16, aoti_torch_dtype_bfloat16)                                  \
  X(kDLBool, 8, aoti_torch_dtype_bool)                                         \
  X(kDLComplex, 32, aoti_torch_dtype_complex32)                                \
  X(kDLComplex, 64, aoti_torch_dtype_complex64)                                \
  X(kDLComplex, 128, aoti_torch_dtype_complex128)                              \
  X(kDLFloat, 16, aoti_torch_dtype_float16)                                    \
  X(kDLFloat, 32, aoti_torch_dtype_float32)                                    \
  X(kDLFloat, 64, aoti_torch_dtype_float64)                                    \
  X(kDLFloat8_e4m3fn, 8, aoti_torch_dtype_float8_e4m3fn)                       \
  X(kDLFloat8_e5m2, 8, aoti_torch_dtype_float8_e5m2)                           \
  X(kDLInt, 8, aoti_torch_dtype_int8)                                          \
  X(kDLInt, 16, aoti_torch_dtype_int16)                                        \
  X(kDLInt, 32, aoti_torch_dtype_int32)                                        \
  X(kDLInt, 64, aoti_torch_dtype_int64)                                        \
  X(kDLUInt, 8, aoti_torch_dtype_uint8)                                        \
  X(kDLUInt, 16, aoti_torch_dtype_uint16)                                      \
  X(kDLUInt, 32, aoti_torch_dtype_uint32)                                      \
  X(kDLUInt, 64, aoti_torch_dtype_uint64)

/// Maps DLDataType (code + bits) to the corresponding PyTorch dtype value.
int32_t mTridentTVMFFIToTorchType(uint8_t dtype_code, uint8_t dtype_bits) {
#define X(dlpack_code, dlpack_bits, torch_fn)                                  \
  if (dtype_code == dlpack_code && dtype_bits == dlpack_bits) {                \
    return torch_fn();                                                         \
  }
  TRIDENT_TVMFFI_DTYPE_PAIR(X)
#undef X
  fprintf(stderr, "Fatal: unhandled DLPack dtype code=%d bits=%d\n",
          (int)dtype_code, (int)dtype_bits);
  abort();
}

// X-Macro: (DLDeviceType, aoti_torch_device_type_* function name)
#define TRIDENT_TVMFFI_DEVICE_PAIR(X)                                          \
  X(kDLCPU, aoti_torch_device_type_cpu)                                        \
  X(kDLCUDA, aoti_torch_device_type_cuda)

/// Maps a DLPack device type to the corresponding PyTorch device type value.
/// Falls back to passthrough (returns the input) for unknown device types.
int32_t mTridentTVMFFIDeviceToTorchDeviceType(int32_t dlDeviceType) {
#define X(dlpack_device, torch_fn)                                             \
  if (dlDeviceType == dlpack_device) {                                         \
    return torch_fn();                                                         \
  }
  TRIDENT_TVMFFI_DEVICE_PAIR(X)
#undef X
  fprintf(stderr, "Fatal: unhandled DLPack device type=%d\n",
          (int)dlDeviceType);
  abort();
}

/// Reverse mapping: Torch dtype → DLPack DLDataType.
/// Uses the same X-Macro pairs as the forward mapping.
DLDataType mTridentTorchToTVMFFIDtype(int32_t torch_dtype) {
#define X(dlpack_code, dlpack_bits, torch_fn)                                  \
  if (torch_dtype == torch_fn()) {                                             \
    return DLDataType{dlpack_code, dlpack_bits, /*lanes=*/1};                  \
  }
  TRIDENT_TVMFFI_DTYPE_PAIR(X)
#undef X
  // Fallback: kDLFloat / 32 / lanes=1.
  return DLDataType{kDLFloat, 32, 1};
}

/// Reverse mapping: Torch device type → DLPack device type.
/// Uses the same X-Macro pairs as the forward mapping.
DLDeviceType mTridentTorchToTVMFFIDevice(int32_t torch_device_type) {
#define X(dlpack_device, torch_fn)                                             \
  if (torch_device_type == torch_fn()) {                                       \
    return dlpack_device;                                                      \
  }
  TRIDENT_TVMFFI_DEVICE_PAIR(X)
#undef X
  // Fallback: kDLCPU.
  return kDLCPU;
}

/// Pack an AtenTensorHandle into a TVMFFIObjectHandle as a kTVMFFITensor
/// object.
///
/// Flow:
/// 1. Extract tensor properties via aoti_torch_get_* functions.
/// 2. Reverse-map Torch dtype/device → DLPack.
/// 3. Heap-allocate a DLManagedTensor, fill with extracted properties.
/// 4. Convert to TVMFFIObjectHandle via TVMFFITensorFromDLPack.
/// 5. Store the handle in *out_handle.
///
/// \return 0 on success, non-zero on failure.
TRIDENT_CORE_RUNTIME_EXPORT int32_t mTridentTensorToTVMFFIObject(
    AtenTensorHandle input, TVMFFIObjectHandle *out_handle) {
  // Step 1: Extract tensor properties via aoti_torch_get_*.
  int64_t ndim_i64;
  int64_t *sizes_ptr;
  int64_t *strides_ptr;
  void *data_ptr;
  int32_t torch_dtype;
  int32_t torch_device_type;
  int32_t device_index;
  int64_t storage_offset;

  TRIDENT_TVMFFI_CHECK(aoti_torch_get_dim(input, &ndim_i64));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_sizes(input, &sizes_ptr));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_strides(input, &strides_ptr));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_data_ptr(input, &data_ptr));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_dtype(input, &torch_dtype));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_device_type(input, &torch_device_type));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_device_index(input, &device_index));
  TRIDENT_TVMFFI_CHECK(aoti_torch_get_storage_offset(input, &storage_offset));

  // Step 2: Reverse-map Torch dtype/device → DLPack.
  DLDataType dl_dtype = mTridentTorchToTVMFFIDtype(torch_dtype);
  DLDeviceType dl_device = mTridentTorchToTVMFFIDevice(torch_device_type);

  // Step 3: malloc(sizeof(DLManagedTensor)) and fill fields.
  DLManagedTensor *dl_managed =
      static_cast<DLManagedTensor *>(std::malloc(sizeof(DLManagedTensor)));
  TRIDENT_TVMFFI_CHECK(dl_managed == nullptr);

  dl_managed->dl_tensor.data = data_ptr;
  dl_managed->dl_tensor.device = DLDevice{dl_device, device_index};
  dl_managed->dl_tensor.ndim = ndim_i64;
  dl_managed->dl_tensor.dtype = dl_dtype;
  dl_managed->dl_tensor.shape = sizes_ptr;
  dl_managed->dl_tensor.strides = strides_ptr;
  dl_managed->dl_tensor.byte_offset = storage_offset;

  // manager_ctx = AtenTensorHandle.
  dl_managed->manager_ctx = input;

  // deleter = mTridentDLManagedTensorDeleter.
  dl_managed->deleter = mTridentDLManagedTensorDeleter;

  // Step 4: Call TVMFFITensorFromDLPack to create a TVM FFI tensor object.
  TVMFFIObjectHandle obj_handle = nullptr;
  int ret = TVMFFITensorFromDLPack(dl_managed, /*require_alignment=*/0,
                                   /*require_contiguous=*/0, &obj_handle);
  if (ret != 0) {
    // TVMFFITensorFromDLPack takes ownership of dl_managed on success.
    // On failure, we must clean up ourselves.
    mTridentDLManagedTensorDeleter(dl_managed);
    return ret;
  }

  // Step 5: Store the handle directly.
  *out_handle = obj_handle;

  return 0;
}
