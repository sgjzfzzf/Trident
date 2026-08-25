//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "dlpack/dlpack.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"
#include "tvm/ffi/container/tensor.h"
#include "tvm/ffi/reflection/registry.h"
#include <cassert>
#include <cstdlib>
#include <new>
#include <stdexcept>

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

// X-Macro: (DLDeviceType, aoti_torch_device_type_* function name)
#define TRIDENT_TVMFFI_DEVICE_PAIR(X)                                          \
  X(kDLCPU, aoti_torch_device_type_cpu)                                        \
  X(kDLCUDA, aoti_torch_device_type_cuda)

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("trident.runtime.tvm_ffi_to_torch_type",
           [](DLDataType dtype) -> int32_t {
#define X(dlpack_code, dlpack_bits, torch_fn)                                  \
  if (dtype.code == dlpack_code && dtype.bits == dlpack_bits) {                \
    return torch_fn();                                                         \
  }
             TRIDENT_TVMFFI_DTYPE_PAIR(X)
#undef X
             return -1;
           })
      .def("trident.runtime.tvm_ffi_device_to_torch_device_type",
           [](int32_t dl_device_type) -> int32_t {
#define X(dlpack_device, torch_fn)                                             \
  if (dl_device_type == dlpack_device) {                                       \
    return torch_fn();                                                         \
  }
             TRIDENT_TVMFFI_DEVICE_PAIR(X)
#undef X
             throw std::runtime_error("unsupported DLPack device type");
           })
      .def("trident.runtime.tensor_to_tvm_ffi_object",
           [](void *input_ptr) -> tvm::ffi::Tensor {
             AtenTensorHandle input = static_cast<AtenTensorHandle>(input_ptr);
             int64_t ndim;
             int64_t *sizes;
             int64_t *strides;
             void *data;
             int32_t torch_dtype;
             int32_t torch_device_type;
             int32_t device_index;
             int64_t storage_offset;
             int status =
                 aoti_torch_get_dim(input, &ndim) |
                 aoti_torch_get_sizes(input, &sizes) |
                 aoti_torch_get_strides(input, &strides) |
                 aoti_torch_get_data_ptr(input, &data) |
                 aoti_torch_get_dtype(input, &torch_dtype) |
                 aoti_torch_get_device_type(input, &torch_device_type) |
                 aoti_torch_get_device_index(input, &device_index) |
                 aoti_torch_get_storage_offset(input, &storage_offset);
             assert(status == 0);
             DLDataType dl_dtype{};
             bool dtype_matched = false;
#define X(dlpack_code, dlpack_bits, torch_fn)                                  \
  if (torch_dtype == torch_fn()) {                                             \
    dl_dtype = DLDataType{dlpack_code, dlpack_bits, /*lanes=*/1};              \
    dtype_matched = true;                                                      \
  }
             TRIDENT_TVMFFI_DTYPE_PAIR(X)
#undef X
             if (!dtype_matched) {
               dl_dtype = DLDataType{kDLFloat, 32, 1};
             }

             DLDeviceType dl_device = kDLCPU;
             bool device_matched = false;
#define X(dlpack_device, torch_fn)                                             \
  if (torch_device_type == torch_fn()) {                                       \
    dl_device = dlpack_device;                                                 \
    device_matched = true;                                                     \
  }
             TRIDENT_TVMFFI_DEVICE_PAIR(X)
#undef X
             if (!device_matched) {
               dl_device = kDLCPU;
             }

             DLManagedTensor *managed = static_cast<DLManagedTensor *>(
                 std::malloc(sizeof(DLManagedTensor)));
             if (managed == nullptr) {
               throw std::bad_alloc();
             }
             managed->dl_tensor.data = data;
             managed->dl_tensor.device = DLDevice{dl_device, device_index};
             managed->dl_tensor.ndim = ndim;
             managed->dl_tensor.dtype = dl_dtype;
             managed->dl_tensor.shape = sizes;
             managed->dl_tensor.strides = strides;
             managed->dl_tensor.byte_offset = storage_offset;
             managed->manager_ctx = input;
             managed->deleter = [](DLManagedTensor *self) {
               if (self != nullptr) {
                 if (self->manager_ctx != nullptr) {
                   aoti_torch_delete_tensor_object(
                       static_cast<AtenTensorHandle>(self->manager_ctx));
                 }
                 std::free(self);
               }
             };
             return tvm::ffi::Tensor::FromDLPack(managed);
           });
}

#undef TRIDENT_TVMFFI_DTYPE_PAIR
#undef TRIDENT_TVMFFI_DEVICE_PAIR
