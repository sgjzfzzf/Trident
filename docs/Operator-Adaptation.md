<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Operator adaptation

FX is imported directly to semantic TVM FFI and native MLIR operations. Start
with the captured FX graph and the dispatch schema, then choose the layer that
owns the behavior. Read [Architecture.md](Architecture.md) before changing the
import or lowering pipeline.

## ATen operators

`ffi/lib/aten/atengen.py` enumerates dispatcher schemas and generates registered
`trident.aten.<name>[.<overload>]` wrappers. Do not edit generated wrappers.
`TridentFxImporter._aten` orders arguments from the schema, fills defaults and
emits a checked `tvm_ffi.FunctionCall`. Most ATen operators therefore need no
individual import pattern. Tensor aliases and mutations follow the boxed ATen
call directly; the frontend does not functionalize the graph.

Check membership in `torch._C._dispatch_get_all_op_names()` before emitting a
registry call. A JIT schema alone does not establish dispatch support. Scalar
shape arithmetic belongs in native SSA, including comparisons, floor division,
remainder and boolean operations. Metadata reads use `tvm_ffi.tensor.size`,
`stride`, `dim`, `device`, `dtype` or `storage_offset`. Dtype values must be
converted to ATen's integer enum when the schema expects it.

For a new operator:

1. Inspect the raw exported graph and schema, including defaults and overloads.
2. Verify the generated wrapper exists, or add an import/lowering rule for an
   operation that has no dispatch schema.
3. Add an importer or end-to-end Python test under `test/` and a focused MLIR
   test at the relevant dialect, conversion or pipeline layer.
4. Run the focused tests, full Python suite and core Lit target.

When adapting a mutation, verify both input writeback and output storage alias
on a cached call. When adapting a factory or clone, verify dtype/device and
allocation behavior. Multi-result calls return an FFI Array; retrieve each
result with its semantic type. Python output container structure is restored
from its pytree specification.

## Semantic IR and ABI

FFI operations accept semantic FFI operands. Use `tvm_ffi.to` for native scalar
boxing, `tvm_ffi.get` for scalar extraction, and `tvm_ffi.cast` for widening an ABI
value to Any or a compatible union. Do not introduce frontend types into these
operations. Every checked registry call must assert lookup status, non-null
handle and call status before using its result. Shared Python builders live in
`python/trident/ir_utils.py`.

Ownership is handled by operation interfaces and the deallocation pass before
ABI lowering. Mark an allocation, borrow, alias or transfer accurately rather
than inserting reference-count calls in the importer. Extend semantic operations
and their lowering under `core/include/trident/core/Dialect/TVMFFI/` and
`core/lib/Conversion/` when an existing operation cannot represent the behavior.
Prefer DRR, then PDLL, for ordinary rewrites; use a C++ conversion pattern for
legality, type conversion or materialization requirements.

When debugging native calls, inspect LLVM's cached `TVMFFIFunctionGetGlobal`
names and compare them with the runtime registry. Calling a non-dispatch scalar
schema or supplying a dtype object to an integer schema can fail at the ABI
boundary even if import and IR verification succeeded.

## Symbolic shapes

Bind tensor dimensions and exported scalar inputs to runtime SSA values.
Evaluate expressions from these bindings; do not turn a SymInt hint into a
runtime constant. Add tests that reuse one compiled specialization with different
shapes and check both IR data flow and numerical results. Unsupported expressions
must report an import error. Triton's binder may use specialization hints to
select a kernel, but launch dimensions and runtime operands must use SSA values.

## Triton and kernel launch adaptation

`python/trident/triton_importer.py` reuses the compiled Triton cache after warmup,
imports `gpu.binary`, and emits `torchext.trident_kernel_launch`. Each source
parameter has a constant or variable specialization attribute. The latter
records native ABI type and optional divisibility; constexpr operands are checked
and omitted from the native kernel call. Functional wrapper outputs clone only
the specified tensors before launch.

`ConvertTorchExtToGPU` runs after inlining and before ownership deallocation.
It extracts native tensor data pointers and scalars from semantic operands,
validates specializations, obtains the current stream through TVM FFI, and emits
`gpu.launch_func`. A specialization failure must return the existing GuardMatch
exception through the enclosing wrapper's Any/union result.

Add a kernel launch or specialization Lit test and a cached Python execution
test for changes to this path. Use hook-free `triton.Config` definitions in JIT
examples because Dynamo export cannot capture arbitrary Triton callbacks.
