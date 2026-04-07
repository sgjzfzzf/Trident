/// e2e-dlpack-runner.c
/// Loads a TVM FFI compiled kernel SO and executes a wrapper via stable C ABI.
/// Usage: e2e-dlpack-runner <kernel.so> [test_name]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tvm/ffi/c_api.h"

static TVMFFIObjectHandle fn_load_module = NULL;
static TVMFFIObjectHandle fn_get_function = NULL;

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

// Test: add_kernel(a, b, c) where c = a + b
static int test_add_kernel(const char *kernel_path) {
  int ret_code = 0;
  TVMFFIAny result_any = {.type_index = kTVMFFINone};
  TVMFFIAny call_args[3] = {};
  TVMFFIAny mod = {.type_index = kTVMFFINone, .v_obj = NULL};
  TVMFFIAny func = {.type_index = kTVMFFINone, .v_obj = NULL};

  printf("[TEST] add_kernel\n");

  // Step 1. Load module from .so
  call_args[0] =
      (TVMFFIAny){.type_index = kTVMFFIRawStr, .v_c_str = kernel_path};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFISmallStr, .v_int64 = 0};
  if ((ret_code = TVMFFIFunctionCall(fn_load_module, call_args, 2, &mod)))
    goto _RAII;

  // Step 2. Get wrapper function from module
  call_args[0] = (TVMFFIAny){.type_index = mod.type_index, .v_obj = mod.v_obj};
  call_args[1] =
      (TVMFFIAny){.type_index = kTVMFFIRawStr, .v_c_str = "add_kernel"};
  call_args[2] = (TVMFFIAny){.type_index = kTVMFFIBool, .v_int64 = 0};
  if ((ret_code = TVMFFIFunctionCall(fn_get_function, call_args, 3, &func)))
    goto _RAII;
  if (func.type_index < kTVMFFIObject || func.v_obj == NULL) {
    fprintf(stderr, "ERROR: module function add_kernel not found\n");
    ret_code = -1;
    goto _RAII;
  }

  // Step 3. Call wrapper function
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
  if (mod.type_index >= kTVMFFIObject)
    TVMFFIObjectDecRef(mod.v_obj);
  if (func.type_index >= kTVMFFIObject)
    TVMFFIObjectDecRef(func.v_obj);
  if (result_any.type_index >= kTVMFFIObject)
    TVMFFIObjectDecRef(result_any.v_obj);
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
  const char *test_name = (argc > 2) ? argv[2] : "add_kernel";

  if ((result = Initialize())) {
    Finalize(result);
    return result;
  }

  if (strcmp(test_name, "add_kernel") == 0 || strcmp(test_name, "all") == 0) {
    result |= test_add_kernel(kernel_path);
  }

  if (result == 0) {
    printf("\n=== ALL TESTS PASSED ===\n");
  } else {
    printf("\n=== SOME TESTS FAILED ===\n");
  }

  Finalize(result);
  return result;
}
