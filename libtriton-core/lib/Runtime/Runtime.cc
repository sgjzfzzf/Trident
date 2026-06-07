#include "libtriton-core/Runtime/Runtime.h"
#include "dlpack/dlpack.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"

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
