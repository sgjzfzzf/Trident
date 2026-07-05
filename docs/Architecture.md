# Trident Architecture

This document describes the core workflows in the current repository:

- Build workflow: how the Python packaging entry points trigger CMake and external dependency builds.
- Runtime workflow: how a Python function is transformed into MLIR/LLVM and then executed via JIT.

## 1. High-Level Components

- Top-level CMake project
  - Dependencies are orchestrated via `ExternalProject` in the root `CMakeLists.txt`:
    - `llvm-project` (with MLIR + Python bindings enabled)
    - `trident-core` (the core C++/MLIR implementation in this repo)
- trident-core
  - Implements and exports Dialects/Passes/Runtime/Python bindings.
  - Depends on torch-mlir, MLIR, LLVM, CUDAToolkit, Torch, and tvm_ffi.
- Python package trident
  - User-facing entry points: `jit` and `compile`.
  - Core backend object: `TridentGraphModule`.
  - Handles graph export/import, guard specialization, compilation, and execution dispatch.

## 2. Build Workflow

```mermaid
flowchart TD
  A[uv pip install -e . or uv build] --> B[scikit-build-core]
  B --> C[root CMakeLists]
  C --> D[ExternalProject: llvm-project]
  C --> E[ExternalProject: trident-core]
  D --> F[install LLVM/MLIR to build/install/llvm-project]
  F --> E
  E --> G[build and install trident-core to build/install/trident-core]
  G --> H[install Python extension into trident/_C]
```

Build highlights:

- `pyproject.toml` uses `scikit-build-core` as the build backend.
- Build steps fetch and compile `llvm-project` and `torch-mlir`; first builds can take a long time.
- `wheel.install-dir` is configured as `trident/_C`, so Python can import the C++/MLIR bindings directly.

## 3. Runtime Compilation Workflow

Users typically decorate Python functions with `@trident.jit`. The call path is:

```mermaid
flowchart TD
  A["user calls jitted function"] --> B["TridentGraphModule.__call__"]
  B --> C{"existing matching specialization?"}
  C -- "no" --> D["compile(*args)"]
  D --> E["torch._dynamo.export builds FX graph and guards"]
  E --> F["TridentFxImporter imports FX into MLIR"]
  F --> G["create tvm_ffi.func wrapper and attach arg_attrs guards"]
  G --> H["store sub-module"]
  H --> I["stub_compile rebuilds executor"]
  I --> J["merge all sub-modules"]
  J --> K["run trident-lowering-pipeline"]
  K --> L["build LLVM dispatcher"]
  L --> M["ExecutionEngine JIT"]
  M --> N["raw_lookup resolves __tvm_ffi_&lt;fn&gt;"]
  N --> O["wrap as tvm_ffi.Function"]
  O --> B
  C -- "yes" --> P["execute executor directly"]
  P --> Q["return Tensor/container results"]
```

## 4. Specialization And Guard Strategy

Each `compile(*args)` produces a new sub-module with these properties:

- The `main` function symbol is indexed to avoid collisions (for example `main_0`, `main_1`).
- The exported `tvm_ffi.func` is also indexed (for example `<fn>_0`, `<fn>_1`).
- Guards exported by Dynamo are converted to MLIR attributes and attached to `tvm_ffi.func` argument attributes.

When runtime inputs change and guards no longer match:

- A sub-function throws `GuardMatchException` (registered as a tvm_ffi error on the Python side).
- The outer dispatcher checks the error kind and tries the next specialization.
- If none match, it returns a failure code and triggers upper-layer recompilation or failure.

## 5. FX Import And Triton Kernel Handling

During FX import, `TridentGraphNodeImporter` applies custom handling for Triton higher-order ops:

- Retrieves compiled kernels and runtime parameters from Triton JIT/Autotune results.
- Materializes cubin into `gpu.binary` (GPU object) attached to the module.
- Emits `torchext.TridentKernelLaunchOp` to launch kernels.
- For autotune paths, computes/selects launch grids based on `best_config`.

This integrates Triton kernel launches into the MLIR workflow, rather than leaving them as Python-level scheduling only.

## 6. LLVM Dispatcher Semantics

After lowering to LLVM, `TridentGraphModule` generates one unified entry point:

- ABI signature: `i32 (ptr, ptr, i32, ptr)`
- Calls `__tvm_ffi_<fn>_<i>` in order
- Uses `TVMFFIErrorMoveFromRaised` / `TVMFFIErrorSetRaised` to read and restore raised errors
- If error kind is `GuardMatchException`, it continues to the next branch
- If return value is success or error is not a guard miss, it returns immediately

This provides runtime dispatch across multiple specializations under one stable symbol name.

## 7. End-to-End Execution Summary

1. The first call to a Python function triggers compilation.
2. The FX graph is imported into MLIR and wrapped as a tvm_ffi callable.
3. All existing specializations are merged and lowered to LLVM.
4. A dispatcher is generated and JIT-compiled by `ExecutionEngine`.
5. Later calls reuse existing specializations first; guard misses trigger incremental compilation.

This design balances:

- Incremental specialization for dynamic input shapes.
- A unified TVM FFI calling interface.
- A composable compilation path from MLIR pipelines to LLVM.
