/// e2e-dlpack-runner.c
/// Loads a TVM FFI compiled kernel SO and executes a wrapper via stable C ABI.
/// Usage: e2e-dlpack-runner <kernel.so> [test_name]

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tvm/ffi/c_api.h"

static TVMFFIObjectHandle fn_load_module = NULL;
static TVMFFIObjectHandle fn_get_function = NULL;

typedef struct {
  DLManagedTensor managed;
  int64_t shape[1];
  int64_t strides[1];
  int64_t *data;
} OwnedDLManagedTensor;

static int Initialize(void) {
  int ret_code = 0;
  TVMFFIByteArray name_load_module = {.data = "ffi.Module.load_from_file.so",
                                      .size = 28};
  TVMFFIByteArray name_get_function = {.data = "ffi.ModuleGetFunction",
                                       .size = 21};
  if ((ret_code = TVMFFIFunctionGetGlobal(&name_load_module, &fn_load_module)))
    return ret_code;
  if ((ret_code =
           TVMFFIFunctionGetGlobal(&name_get_function, &fn_get_function)))
    return ret_code;
  return 0;
}

static void Finalize(int ret_code) {
  TVMFFIObjectHandle err = NULL;
  if (fn_load_module)
    TVMFFIObjectDecRef(fn_load_module);
  if (fn_get_function)
    TVMFFIObjectDecRef(fn_get_function);
  if (ret_code) {
    TVMFFIErrorCell *cell = NULL;
    TVMFFIErrorMoveFromRaised(&err);
    if (!err)
      return;
    cell = (TVMFFIErrorCell *)((char *)(err) + sizeof(TVMFFIObject));
    printf("%.*s: %.*s\n", (int)(cell->kind.size), cell->kind.data,
           (int)(cell->message.size), cell->message.data);
    TVMFFIObjectDecRef(err);
  }
}

static void DestroyOwnedDLManagedTensor(DLManagedTensor *tensor) {
  OwnedDLManagedTensor *owned = (OwnedDLManagedTensor *)tensor;
  if (owned == NULL) {
    return;
  }
  free(owned->data);
  free(owned);
}

static int CreateOwnedI64Vector(const int64_t *values, int64_t size,
                                DLManagedTensor **out) {
  OwnedDLManagedTensor *owned = NULL;

  if (out == NULL || values == NULL || size <= 0) {
    return -1;
  }

  owned = (OwnedDLManagedTensor *)calloc(1, sizeof(*owned));
  if (owned == NULL) {
    return -1;
  }

  owned->data = (int64_t *)malloc((size_t)size * sizeof(*owned->data));
  if (owned->data == NULL) {
    free(owned);
    return -1;
  }

  memcpy(owned->data, values, (size_t)size * sizeof(*owned->data));
  owned->shape[0] = size;
  owned->strides[0] = 1;
  owned->managed.dl_tensor.data = owned->data;
  owned->managed.dl_tensor.device = (DLDevice){kDLCPU, 0};
  owned->managed.dl_tensor.ndim = 1;
  owned->managed.dl_tensor.dtype = (DLDataType){kDLInt, 64, 1};
  owned->managed.dl_tensor.shape = owned->shape;
  owned->managed.dl_tensor.strides = owned->strides;
  owned->managed.dl_tensor.byte_offset = 0;
  owned->managed.manager_ctx = owned;
  owned->managed.deleter = DestroyOwnedDLManagedTensor;
  *out = &owned->managed;
  return 0;
}

static int CreateTensorArg(const int64_t *values, int64_t size,
                           TVMFFIObjectHandle *out) {
  int ret_code = 0;
  DLManagedTensor *managed = NULL;

  if (CreateOwnedI64Vector(values, size, &managed) != 0) {
    fprintf(stderr, "ERROR: failed to allocate input tensor\n");
    return -1;
  }

  ret_code = TVMFFITensorFromDLPack(managed, 1, 1, out);
  if (ret_code != 0 && managed->deleter != NULL) {
    managed->deleter(managed);
  }
  return ret_code;
}

static void ReleaseAnyObject(TVMFFIAny any) {
  if (any.type_index >= kTVMFFIObject && any.v_obj != NULL) {
    TVMFFIObjectDecRef(any.v_obj);
  }
}

static int LoadKernelFunction(const char *kernel_path,
                              const char *function_name, TVMFFIAny *mod,
                              TVMFFIAny *func) {
  int ret_code = 0;
  TVMFFIAny call_args[3] = {};

  call_args[0] =
      (TVMFFIAny){.type_index = kTVMFFIRawStr, .v_c_str = kernel_path};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFISmallStr, .v_int64 = 0};
  if ((ret_code = TVMFFIFunctionCall(fn_load_module, call_args, 2, mod))) {
    return ret_code;
  }

  call_args[0] =
      (TVMFFIAny){.type_index = mod->type_index, .v_obj = mod->v_obj};
  call_args[1] =
      (TVMFFIAny){.type_index = kTVMFFIRawStr, .v_c_str = function_name};
  call_args[2] = (TVMFFIAny){.type_index = kTVMFFIBool, .v_int64 = 0};
  if ((ret_code = TVMFFIFunctionCall(fn_get_function, call_args, 3, func))) {
    return ret_code;
  }
  if (func->type_index < kTVMFFIObject || func->v_obj == NULL) {
    fprintf(stderr, "ERROR: module function %s not found\n", function_name);
    return -1;
  }
  return 0;
}

static int CheckI64Tensor(DLTensor *tensor, const int64_t *expected,
                          int64_t size) {
  int64_t index = 0;
  int64_t stride = 1;
  int64_t *data = NULL;

  if (tensor == NULL || expected == NULL) {
    fprintf(stderr, "ERROR: null tensor check input\n");
    return -1;
  }
  if (tensor->device.device_type != kDLCPU || tensor->device.device_id != 0) {
    fprintf(stderr, "ERROR: expected CPU tensor, got device=(%d,%d)\n",
            tensor->device.device_type, tensor->device.device_id);
    return -1;
  }
  if (tensor->ndim != 1 || tensor->shape == NULL || tensor->shape[0] != size) {
    fprintf(stderr, "ERROR: expected shape=[%lld], got ndim=%d\n",
            (long long)size, tensor->ndim);
    return -1;
  }
  if (tensor->dtype.code != kDLInt || tensor->dtype.bits != 64 ||
      tensor->dtype.lanes != 1) {
    fprintf(stderr,
            "ERROR: expected dtype=int64, got code=%u bits=%u lanes=%u\n",
            tensor->dtype.code, tensor->dtype.bits, tensor->dtype.lanes);
    return -1;
  }

  stride = tensor->strides == NULL ? 1 : tensor->strides[0];
  data = (int64_t *)((char *)tensor->data + tensor->byte_offset);
  for (index = 0; index < size; ++index) {
    const int64_t actual = data[index * stride];
    if (actual != expected[index]) {
      fprintf(stderr,
              "ERROR: expected result[%lld]=%lld, got %lld (stride=%lld)\n",
              (long long)index, (long long)expected[index], (long long)actual,
              (long long)stride);
      return -1;
    }
  }
  return 0;
}

// Test: scalar_add_kernel(a, b) -> a + b
static int test_scalar_add_kernel(const char *kernel_path) {
  int ret_code = 0;
  TVMFFIAny result_any = {.type_index = kTVMFFINone};
  TVMFFIAny call_args[2] = {};
  TVMFFIAny mod = {.type_index = kTVMFFINone, .v_obj = NULL};
  TVMFFIAny func = {.type_index = kTVMFFINone, .v_obj = NULL};

  printf("[TEST] scalar_add_kernel\n");

  if ((ret_code =
           LoadKernelFunction(kernel_path, "scalar_add_kernel", &mod, &func))) {
    goto _RAII;
  }

  call_args[0] = (TVMFFIAny){.type_index = kTVMFFIInt, .v_int64 = 40};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFIInt, .v_int64 = 2};
  if ((ret_code = TVMFFIFunctionCall((TVMFFIObjectHandle)func.v_obj, call_args,
                                     2, &result_any)))
    goto _RAII;

  if (result_any.type_index != kTVMFFIInt || result_any.v_int64 != 42) {
    fprintf(stderr, "ERROR: expected int(42), got type=%d value=%lld\n",
            result_any.type_index, (long long)result_any.v_int64);
    ret_code = -1;
    goto _RAII;
  }
  printf("  [OK] Output correct: 40 + 2 = %lld\n",
         (long long)result_any.v_int64);
  ret_code = 0;

_RAII:
  ReleaseAnyObject(mod);
  ReleaseAnyObject(func);
  ReleaseAnyObject(result_any);
  return ret_code;
}

// Test: tensor_add_kernel(x, y) returns x + y on CPU.
static int test_tensor_add_kernel(const char *kernel_path) {
  static const int64_t lhs_values[4] = {1, 2, 3, 4};
  static const int64_t rhs_values[4] = {10, 20, 30, 40};
  static const int64_t expected[4] = {11, 22, 33, 44};

  int ret_code = 0;
  TVMFFIObjectHandle lhs_tensor = NULL;
  TVMFFIObjectHandle rhs_tensor = NULL;
  TVMFFIAny result_any = {.type_index = kTVMFFINone};
  TVMFFIAny call_args[2] = {};
  TVMFFIAny mod = {.type_index = kTVMFFINone, .v_obj = NULL};
  TVMFFIAny func = {.type_index = kTVMFFINone, .v_obj = NULL};
  DLManagedTensor *result_managed = NULL;

  printf("[TEST] tensor_add_kernel\n");

  if ((ret_code = CreateTensorArg(lhs_values, 4, &lhs_tensor))) {
    goto _RAII;
  }
  if ((ret_code = CreateTensorArg(rhs_values, 4, &rhs_tensor))) {
    goto _RAII;
  }

  if ((ret_code =
           LoadKernelFunction(kernel_path, "tensor_add_kernel", &mod, &func))) {
    goto _RAII;
  }

  call_args[0] = (TVMFFIAny){.type_index = kTVMFFITensor,
                             .v_obj = (TVMFFIObject *)lhs_tensor};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFITensor,
                             .v_obj = (TVMFFIObject *)rhs_tensor};
  if ((ret_code = TVMFFIFunctionCall((TVMFFIObjectHandle)func.v_obj, call_args,
                                     2, &result_any))) {
    goto _RAII;
  }

  if (result_any.type_index != kTVMFFITensor || result_any.v_obj == NULL) {
    fprintf(stderr, "ERROR: expected tensor return value, got type=%d\n",
            result_any.type_index);
    ret_code = -1;
    goto _RAII;
  }

  if ((ret_code = TVMFFITensorToDLPack((TVMFFIObjectHandle)result_any.v_obj,
                                       &result_managed))) {
    goto _RAII;
  }
  if ((ret_code = CheckI64Tensor(&result_managed->dl_tensor, expected, 4))) {
    goto _RAII;
  }

  printf("  [OK] Output correct: [1, 2, 3, 4] + [10, 20, 30, 40] = "
         "[11, 22, 33, 44]\n");

_RAII:
  if (result_managed != NULL && result_managed->deleter != NULL) {
    result_managed->deleter(result_managed);
    result_managed = NULL;
  }
  ReleaseAnyObject(result_any);
  ReleaseAnyObject(mod);
  ReleaseAnyObject(func);
  if (lhs_tensor != NULL) {
    TVMFFIObjectDecRef(lhs_tensor);
  }
  if (rhs_tensor != NULL) {
    TVMFFIObjectDecRef(rhs_tensor);
  }
  return ret_code;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <kernel.so> [test_name]\n", argv[0]);
    return 1;
  }

  const char *kernel_path = argv[1];
  printf("Loading kernel: %s\n", kernel_path);

  int result = 0;
  const char *test_name = (argc > 2) ? argv[2] : "all";

  if ((result = Initialize())) {
    Finalize(result);
    return result;
  }

  if (strcmp(test_name, "scalar_add_kernel") == 0 ||
      strcmp(test_name, "all") == 0) {
    result |= test_scalar_add_kernel(kernel_path);
  }
  if (strcmp(test_name, "tensor_add_kernel") == 0 ||
      strcmp(test_name, "all") == 0) {
    result |= test_tensor_add_kernel(kernel_path);
  }

  if (result == 0) {
    printf("\n=== ALL TESTS PASSED ===\n");
  } else {
    printf("\n=== SOME TESTS FAILED ===\n");
  }

  Finalize(result);
  return result;
}
