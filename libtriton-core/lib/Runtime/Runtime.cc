#include "libtriton-core/Runtime/Runtime.h"
#include "dlpack/dlpack.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"

#include <cstdlib>

/// Check that a call returns 0 (success), or that a condition is false;
/// return -1 on failure.
#define LIBTRITON_TVMFFI_CHECK(cond)                                           \
  do {                                                                         \
    if (cond) {                                                                \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// X-Macro: (DLDataTypeCode, bits, aoti_torch_dtype_* function name)
#define LIBTRITON_TVMFFI_DTYPE_PAIR(X)                                         \
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
int32_t mLibTritonTVMFFIToTorchType(uint8_t dtype_code, uint8_t dtype_bits) {
#define X(dlpack_code, dlpack_bits, torch_fn)                                  \
  if (dtype_code == static_cast<uint8_t>(dlpack_code) &&                       \
      dtype_bits == static_cast<uint8_t>(dlpack_bits)) {                       \
    return torch_fn();                                                         \
  }
  LIBTRITON_TVMFFI_DTYPE_PAIR(X)
#undef X
  fprintf(stderr, "Fatal: unhandled DLPack dtype code=%d bits=%d\n",
          (int)dtype_code, (int)dtype_bits);
  abort();
}

// X-Macro: (DLDeviceType, aoti_torch_device_type_* function name)
#define LIBTRITON_TVMFFI_DEVICE_PAIR(X)                                        \
  X(kDLCPU, aoti_torch_device_type_cpu)                                        \
  X(kDLCUDA, aoti_torch_device_type_cuda)

/// Maps a DLPack device type to the corresponding PyTorch device type value.
/// Falls back to passthrough (returns the input) for unknown device types.
int32_t mLibTritonTVMFFIDeviceToTorchDeviceType(int32_t dlDeviceType) {
#define X(dlpack_device, torch_fn)                                             \
  if (dlDeviceType == static_cast<int32_t>(dlpack_device)) {                   \
    return torch_fn();                                                         \
  }
  LIBTRITON_TVMFFI_DEVICE_PAIR(X)
#undef X
  fprintf(stderr, "Fatal: unhandled DLPack device type=%d\n",
          (int)dlDeviceType);
  abort();
}

/// Reverse mapping: Torch dtype → DLPack DLDataType.
/// Uses the same X-Macro pairs as the forward mapping.
DLDataType mLibTritonTorchToTVMFFIDtype(int32_t torch_dtype) {
#define X(dlpack_code, dlpack_bits, torch_fn)                                  \
  if (torch_dtype == torch_fn()) {                                             \
    return DLDataType{static_cast<uint8_t>(dlpack_code),                       \
                      static_cast<uint8_t>(dlpack_bits), /*lanes=*/1};         \
  }
  LIBTRITON_TVMFFI_DTYPE_PAIR(X)
#undef X
  // Fallback: kDLFloat / 32 / lanes=1.
  return DLDataType{static_cast<uint8_t>(kDLFloat), 32, 1};
}

/// Reverse mapping: Torch device type → DLPack device type.
/// Uses the same X-Macro pairs as the forward mapping.
DLDeviceType mLibTritonTorchToTVMFFIDevice(int32_t torch_device_type) {
#define X(dlpack_device, torch_fn)                                             \
  if (torch_device_type == torch_fn()) {                                       \
    return dlpack_device;                                                      \
  }
  LIBTRITON_TVMFFI_DEVICE_PAIR(X)
#undef X
  // Fallback: kDLCPU.
  return kDLCPU;
}

/// DLPack-compatible deleter callback for TVM FFI tensor conversion.
///
/// Extracts the AtenTensorHandle stored in self->manager_ctx, deletes it via
/// aoti_torch_delete_tensor_object, and frees the DLManagedTensor allocation.
extern "C" void mLibTritonDLManagedTensorDeleter(struct DLManagedTensor *self) {
  if (self) {
    if (self->manager_ctx) {
      aoti_torch_delete_tensor_object(
          static_cast<AtenTensorHandle>(self->manager_ctx));
    }
    std::free(self);
  }
}

/// Pack an AtenTensorHandle into a TVMFFIAny slot as a kTVMFFITensor object.
///
/// Flow:
/// 1. Extract tensor properties via aoti_torch_get_* functions.
/// 2. Reverse-map Torch dtype/device → DLPack.
/// 3. Heap-allocate a DLManagedTensor, fill with extracted properties.
/// 4. Convert to TVMFFIObjectHandle via TVMFFITensorFromDLPack.
/// 5. Store type_index=kTVMFFITensor(70) and v_obj=handle in TVMFFIAny.
///
/// \return 0 on success, non-zero on failure.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t
mLibTritonPackTensorToTVMFFIAny(AtenTensorHandle input, TVMFFIAny *ptr) {
  // Step 1: Extract tensor properties via aoti_torch_get_*.
  int64_t ndim_i64;
  int64_t *sizes_ptr;
  int64_t *strides_ptr;
  void *data_ptr;
  int32_t torch_dtype;
  int32_t torch_device_type;
  int32_t device_index;
  int64_t storage_offset;

  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_dim(input, &ndim_i64));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_sizes(input, &sizes_ptr));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_strides(input, &strides_ptr));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_data_ptr(input, &data_ptr));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_dtype(input, &torch_dtype));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_device_type(input, &torch_device_type));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_device_index(input, &device_index));
  LIBTRITON_TVMFFI_CHECK(aoti_torch_get_storage_offset(input, &storage_offset));

  // Step 2: Reverse-map Torch dtype/device → DLPack.
  DLDataType dl_dtype = mLibTritonTorchToTVMFFIDtype(torch_dtype);
  DLDeviceType dl_device = mLibTritonTorchToTVMFFIDevice(torch_device_type);

  // Step 3: malloc(sizeof(DLManagedTensor)) and fill fields.
  DLManagedTensor *dl_managed =
      static_cast<DLManagedTensor *>(std::malloc(sizeof(DLManagedTensor)));
  LIBTRITON_TVMFFI_CHECK(dl_managed == nullptr);

  dl_managed->dl_tensor.data = data_ptr;
  dl_managed->dl_tensor.device = DLDevice{dl_device, device_index};
  dl_managed->dl_tensor.ndim = static_cast<int32_t>(ndim_i64);
  dl_managed->dl_tensor.dtype = dl_dtype;
  dl_managed->dl_tensor.shape = sizes_ptr;
  dl_managed->dl_tensor.strides = strides_ptr;
  dl_managed->dl_tensor.byte_offset = static_cast<uint64_t>(storage_offset);

  // manager_ctx = AtenTensorHandle.
  dl_managed->manager_ctx = input;

  // deleter = mLibTritonDLManagedTensorDeleter.
  dl_managed->deleter = mLibTritonDLManagedTensorDeleter;

  // Step 4: Call TVMFFITensorFromDLPack to create a TVM FFI tensor object.
  TVMFFIObjectHandle obj_handle = nullptr;
  int ret = TVMFFITensorFromDLPack(dl_managed, /*require_alignment=*/0,
                                   /*require_contiguous=*/0, &obj_handle);
  if (ret != 0) {
    // TVMFFITensorFromDLPack takes ownership of dl_managed on success.
    // On failure, we must clean up ourselves.
    mLibTritonDLManagedTensorDeleter(dl_managed);
    return ret;
  }

  // Step 5: Store into TVMFFIAny: type_index=kTVMFFITensor, payload=v_obj.
  ptr->type_index = kTVMFFITensor;
  ptr->zero_padding = 0;
  ptr->v_obj = static_cast<TVMFFIObject *>(obj_handle);

  return 0;
}

/// Unpack a TVMFFIAny slot into an AtenTensorHandle.
///
/// The payload may be:
/// - kTVMFFIDLTensorPtr (7): direct DLTensor* pointer in v_ptr.
/// - kTVMFFITensor (70) or other object types: TVMFFIObject* in v_obj;
///   DLTensor starts at offset sizeof(TVMFFIObject)=24 from the object ptr.
///
/// \return 0 on success, non-zero on failure (e.g. not a tensor type).
LIBTRITON_CORE_RUNTIME_EXPORT int32_t mLibTritonUnpackTVMFFIAnyToTensor(
    const TVMFFIAny *ptr, AtenTensorHandle *output) {
  DLTensor *dl_tensor = nullptr;

  if (ptr->type_index == kTVMFFITensor && ptr->v_obj) {
    // TVMFFIObject - DLTensor starts after the object header.
    // TVMFFIObject layout: {uint64_t ref_count, int32_t type_index,
    //                       uint32_t padding, union deleter} = 24 bytes.
    dl_tensor = reinterpret_cast<DLTensor *>(
        reinterpret_cast<uintptr_t>(ptr->v_obj) + sizeof(TVMFFIObject));
  } else if (ptr->type_index == kTVMFFIDLTensorPtr && ptr->v_ptr) {
    // Direct DLTensor* pointer.
    dl_tensor = static_cast<DLTensor *>(ptr->v_ptr);
  } else {
    // Not a tensor type.
    return -1;
  }

  // Map DLPack dtype → Torch dtype.
  int32_t torch_dtype =
      mLibTritonTVMFFIToTorchType(dl_tensor->dtype.code, dl_tensor->dtype.bits);

  // Map DLPack device → Torch device type.
  int32_t torch_device_type =
      mLibTritonTVMFFIDeviceToTorchDeviceType(dl_tensor->device.device_type);

  // Call aoti_torch_create_tensor_from_blob with all extracted fields.
  return aoti_torch_create_tensor_from_blob(
      dl_tensor->data, static_cast<int64_t>(dl_tensor->ndim), dl_tensor->shape,
      dl_tensor->strides, static_cast<int64_t>(dl_tensor->byte_offset),
      torch_dtype, torch_device_type, dl_tensor->device.device_id, output);
}
