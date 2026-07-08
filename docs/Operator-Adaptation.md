# Operator Adaptation Guide

This guide explains how to adapt a new operator in Trident, including:

- What to implement.
- How to validate the implementation.
- What to suspect when bugs appear.

## 1. Pick The Adaptation Path

In this repository, there are two common adaptation paths.

1. Conversion pipeline path (recommended first for a new ATen op)
- You add/update MLIR test inputs under `trident-core/test/Conversion/Pipeline`.
- You verify that `trident-lowering-pipeline` lowers correctly.
- You add a Python end-to-end unittest in `test/` using `AtenOpTest`.

2. Python frontend path (`@trident.jit` dynamic compile)
- You ensure your op appears correctly in exported FX graph.
- You verify FX import and runtime specialization behavior.
- You validate dispatcher behavior across multiple guard specializations.

For a new operator bring-up, start with path 1, then validate path 2 if needed.

## 2. atengen Auto-Wrapping

Most ATen operators are automatically wrapped by the atengen codegen tool at build
time (`trident-core/lib/Runtime/python/atengen.py`). atengen queries PyTorch's JIT
type system via `torch._C._jit_get_all_schemas()` and generates C++ wrappers that
register `trident.aten.*` TVM FFI global functions with proper IValue ↔ TVMFFIAny
conversion.

For a standard ATen operator, **no manual C++ changes are needed**:
- The MLIR lowering (`Aten.cc` -> `ConvertAtenDispatcherOp`) already handles all
  `torch.aten.*` ops generically by rewiring them to `trident.aten.*` FFI calls.
- The atengen-generated wrapper handles IValue ↔ TVMFFIAny conversion automatically
  for all c10 types (TensorType, IntType, FloatType, BoolType, ListType,
  OptionalType, DeviceObjType, NumberType, etc.).

Manual intervention is only needed when:
- The operator requires special lowering beyond the generic `trident.aten.*`
  dispatch (e.g., constant folding, shape-dependent logic).
- The operator uses types not yet supported by atengen's type mapping in
  `Function.h` / `Value.h`.

## 3. Minimal Bring-Up Checklist (Path 1)

1. Add a pipeline test MLIR file
- Create `trident-core/test/Conversion/Pipeline/<op>.mlir`.
- Include one `func.func` using your target op.
- Include one `tvm_ffi.func` wrapper exposing a callable symbol `<op>`.
- Add `// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s`.
- Add focused `FileCheck` assertions for key lowering points.

2. Match wrapper symbol with Python test expectations
- Python tests call `raw_lookup("__tvm_ffi_<op>")` in `test/base.py`.
- Ensure wrapper function is `tvm_ffi.func @<op>(...)` so lowered symbol is consistent.

3. Add Python end-to-end unittest
- Create `test/test_<op>.py`.
- Subclass `AtenOpTest` and return `<op>` in `op_name()`.
- Call `self.ffi_func(...)` with representative inputs.
- Assert output shape/dtype/value semantics.

4. Run tests
- Run operator-specific unittest first, then full suite:
  - `python -m unittest test.test_<op>`
  - `python -m unittest discover -s test -p "test_*.py"`

## 4. Optional Bring-Up Checklist (Path 2)

Use this when the op is exercised through Python functions decorated by `@trident.jit`.

1. Export path sanity
- Confirm `torch._dynamo.export(...)` can capture your function.
- Confirm guards are produced as expected for dynamic shapes/dtypes/devices.

2. FX import sanity
- Ensure the scoped Triton HOP patch (`patch.py`) and node import logic can
  represent the op.
- If Triton higher-order ops are involved, confirm kernel metadata/runtime args
  are materialized correctly.

3. Runtime specialization sanity
- First call should compile a sub-module.
- Subsequent compatible inputs should reuse specialization.
- Guard mismatch should trigger incremental compile, not silent wrong results.

## 5. Common Bug Symptoms And What To Suspect

### A. `raw_lookup("__tvm_ffi_<op>")` returns null / symbol not found

Suspect:
- `tvm_ffi.func` wrapper name mismatch (`@<op>` not matching test `op_name`).
- Wrapper not present in MLIR test file.
- Lowering or symbol merge stage removed/renamed your symbol unexpectedly.

Check:
- `trident-core/test/Conversion/Pipeline/<op>.mlir`
- `test/base.py` symbol construction logic

### B. Pipeline fails before LLVM lowering

Suspect:
- Unsupported op form or operand type in the test IR.
- Missing dialect registration or pass handling for this op.
- Invalid MLIR syntax or wrong op signature in your `.mlir` file.

Check:
- `trident-core/test/Conversion/Pipeline/<op>.mlir`
- `trident-core` dialect/pass implementation touched by your op

### C. Wrapper call succeeds but output shape/dtype is wrong

Suspect:
- Wrapper argument order mismatch.
- Incorrect type conventions between `tvm_ffi.func` signature and op call.
- Device/dtype encoded values are wrong at call site.

Check:
- `test/test_<op>.py` argument construction
- Wrapper body in `<op>.mlir`
- Existing examples: `empty.mlir`, `empty_like.mlir`

### D. Runtime crash around FFI object conversion or ownership

Suspect:
- Wrong tensor/object conversion helper path during lowering.
- Reference count/ownership transfer assumptions violated.
- Returned object type does not match expected TVM FFI kind.

Check:
- Lowered code patterns asserted by `FileCheck`
- Runtime conversion calls in generated LLVM IR path

### E. `@trident.jit` path recompiles too often or never stabilizes

Suspect:
- Guard generation too strict or mismatched with actual call patterns.
- Specialization matching fails due to dtype/device/shape guard differences.
- Dispatcher guard-miss recognition path not classifying errors correctly.

Check:
- Guard parsing and attr generation in `python/trident/guards/`
- Specialization/dispatcher flow in `python/trident/backend.py`
- Error registration for `GuardMatchException` in `python/trident/error.py`

### F. First call works, second call fails with different input

Suspect:
- Incremental sub-module merge issues.
- Incomplete cloning/merge assumptions for previously compiled modules.
- Specialization order-dependent behavior in dispatcher.

Check:
- Combined module build and merge path in `TridentGraphModule`
- Dispatcher branch ordering and return/error handling

### G. Numerical precision/value mismatch appears intermittently

Suspect:
- Guard handling is incorrect, so a stale specialization/cache entry is reused for inputs that should trigger recompilation.
- Guard key dimensions (dtype/device/shape or scalar-kind distinctions) are missing or parsed inconsistently.

Check:
- Guard parsing and matching logic in `python/trident/guards/` and specialization reuse flow in `python/trident/backend.py`.
- Whether the failing case should have produced a guard miss but was incorrectly treated as cache hit.

### H. `torch._dynamo.export` fails on Triton autotune/hook features

Symptom:
- `torch._dynamo.exc.Unsupported` reports Triton kernel unsupported features, especially `pre_hook`/`post_hook` on `triton.Config` or `triton.autotune`.

Cause:
- Torch Dynamo does not fully support tracing/exporting Triton hook callbacks in the JIT/export path.

Recommendation:
- Avoid Triton hooks (`pre_hook`/`post_hook`) in code paths that may run under `@trident.jit` or `torch._dynamo.export`.
- Prefer hook-free `triton.Config(...)` definitions and keep JIT-facing wrappers on Torch-friendly code paths.

## 6. Practical Debug Strategy

1. Start from smallest reproducible input.
2. Verify pipeline-only behavior with `<op>.mlir` + `FileCheck` first.
3. Verify end-to-end unittest with `AtenOpTest`.
4. Only then debug `@trident.jit` dynamic specialization behavior.
5. If failure is unclear, compare your new op with known-good patterns:
- `trident-core/test/Conversion/Pipeline/empty.mlir`
- `trident-core/test/Conversion/Pipeline/empty_like.mlir`
- `test/test_empty.py`
- `test/test_empty_like.py`
6. Prefer functional-style graph rewrites during debugging; avoid in-place tensor mutation when possible, because in-place ops can hide data-flow issues and complicate guard/cache correctness analysis.

## 7. Definition Of Done

An operator adaptation is considered done when:

- Pipeline test exists and passes with stable `FileCheck` assertions.
- Python end-to-end unittest exists and passes.
- Symbol naming and wrapper ABI are consistent.
- No unexpected recompilation or guard-dispatch regression appears in runtime path.
