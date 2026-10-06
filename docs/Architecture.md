<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Architecture

Trident captures Python tensor programs with Dynamo and `torch.export`, imports
FX directly into MLIR, and executes the lowered program through TVM FFI. The
frontend owns the import; the build depends on LLVM/MLIR, PyTorch, CUDA and TVM
FFI. The LLVM revision remains pinned in the top-level CMake file.

## Frontend and symbolic values

`python/trident/backend.py` captures a graph and its Dynamo guards. It exports
without running functionalization or decompositions, executes a warmup to populate
Triton's compilation cache, and calls `TridentFxImporter.import_program`.
Warmup is the first execution, including its mutations; cached calls execute the
compiled program. A private `func.func` holds the graph, and a public
`tvm_ffi.func` checks guards and exposes the packed ABI. The Python executor
converts runtime tensors through DLPack and reconstructs the output pytree.

`fx_importer.py` maps FX nodes to SSA values. Tensors use `!tvm_ffi.tensor` and
retain reference semantics: an in-place ATen call operates on the original
object, and views keep their storage aliases. ATen calls use the schema to
order positional and keyword arguments, fill defaults, select a scalar overload
when FX recorded a Python scalar in a tensor overload, and name the existing
`trident.aten.<operator>[.<overload>]` registry wrapper. Dtype arguments pass
through `trident.runtime.tvm_ffi_to_torch_type` to obtain the integer expected by
ATen's boxed schema. Multiple ATen results are retrieved from an FFI Array.

SymInt, SymFloat and SymBool computations use native `i64`, `f64` and `i1` SSA.
Input tensor dimensions bind symbols to `tvm_ffi.tensor.size`; explicit scalar
nodes bind their expressions to imported SSA values. Composite shape expressions
are evaluated from those bindings. Runtime expressions never use example hints;
unbound symbols produce an explicit import error. Native scalars are boxed only
at FFI calls, containers or function return boundaries. `torch.cond` imports its
branches into `scf.if`, with native predicates and semantic result values.

`ir_utils.py` shares scalar arithmetic, constants, boxing and checked calls with
the guard frontend. Each call asserts lookup success, a non-null handle and call
success. Integer division uses floor semantics; integer remainder follows
`a - floor(a / b) * b`. Supported arithmetic follows the runtime's 64-bit scalar
ABI. Unsupported scalar expressions or higher-order operators fail explicitly.

## Guards and containers

Guards are parsed into `Code`, `Guard` and collection objects under
`python/trident/guards/`. `InputTableBuilder` maps function arguments and nested
list/tuple paths to semantic values. Containers use `!tvm_ffi.array`; extraction
calls `ffi.ArrayGetItem` lazily in the block that needs the element. Scalar guards
use native arithmetic and comparisons; tensor metadata uses existing semantic
metadata operations. Guard collections build short-circuit CFG branches. A
failed specialization returns `!tvm_ffi.exception` widened to `!tvm_ffi.any`, so
the dispatcher can try another specialization or request compilation.

Tensor identity and `_base` guards retain their existing conservative handling;
DLPack does not expose Python wrapper identity or autograd view relationships.

## Triton kernels

`triton_importer.py` resolves the compiled Triton kernel from its warmup cache,
imports its cubin as `gpu.binary`, and emits `tvm_ffi.kernel_launch`.
Each specialization has a unique binary symbol. Operands retain semantic FFI
values and carry constant or variable specialization attributes in source
parameter order. Variable attributes record native ABI type and divisibility;
constant attributes include scalar, string and nested tuple values. Binder hints
are used only to select the compiled kernel, never to substitute runtime shapes.
Functional Triton wrapper outputs explicitly clone `tensors_to_clone` before
launching; mutation wrappers use the original operands.

`ConvertTVMFFIToGPU` validates specialization attributes, branches to
`GuardMatch` on failure, extracts tensor pointers and scalar values, and emits
`gpu.launch_func` with the current TVM FFI CUDA stream. Constant operands are
checked and omitted from the native kernel argument list. The ordinary inliner
runs first so these failure branches belong to the public guarded wrapper.

## Lowering and ownership

The lowering pipeline is:

1. Inline private graph functions and lower kernel launches to GPU operations.
2. Lower SCF to CFG and finalize semantic metadata operations.
3. Insert ownership deallocation while semantic object types are available.
4. Convert `tvm_ffi.func` to ordinary functions and packed ABI wrappers.
5. Lower semantic values and DLPack to LLVM, followed by native arithmetic,
   control flow and GPU conversions, canonicalization and cast reconciliation.

The TVM FFI dialect distinguishes boxed values, owned runtime objects, borrowed
objects and native pointers. Its ownership interfaces describe allocations,
borrows, transfers and aliases; the ownership pass retains escaping borrowed
values and releases owned values along CFG edges. The importer does not add
manual reference counting. Runtime objects use the TVMFFIAny representation
`!llvm.struct<(i32, i32, i64)>`; function handles are native pointers.

`tvm_ffi.cast` widens a semantic ABI value to Any or a compatible union;
`tvm_ffi.get` extracts a native scalar or borrowed object, and `tvm_ffi.to` boxes
a native scalar. Metadata operations lower through DLPack. Tensor literals use
the existing runtime staging path. Global function handles are cached at module
initialization; call sites retain and release the cached reference.

## Source layout and validation

`core/` contains the DLPack and TVM FFI dialects, lowering passes,
C API and Python bindings. `ffi/` generates ATen wrappers from dispatcher schemas
and provides runtime conversion and exception helpers. Python import and guard
tests live under `test/`; dialect, conversion and pipeline tests live under
`core/test/`. Operator work follows [Operator-Adaptation.md](Operator-Adaptation.md).

Build with the editable command in the repository contributor guide, then run
`python -m unittest discover -s test -p 'test_*.py'` and
`cmake --build build/core --target check-trident-core`. A standalone core build
uses an installed MLIR CMake package; it does not fetch another compiler frontend.
CUDA is required by GPU execution tests.
