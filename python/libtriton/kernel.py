from __future__ import annotations

from functools import cached_property
from typing import Any, Final, List, Tuple, TYPE_CHECKING

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, func, gpu, llvm

if TYPE_CHECKING:
    import triton


_LLVM_GEP_DYNAMIC_INDEX: Final[int] = -2147483648


class KernelBuilder:
    """Builder for constructing MLIR GPU launch modules from Triton compiled kernels."""

    def __init__(self, compiled_kernel: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_kernel: Final[triton.compiler.CompiledKernel] = compiled_kernel

    @cached_property
    def parameters(self) -> List[Tuple[str, str]]:
        """Get all kernel parameters (name, Triton type string) from compiled kernel signature."""
        return list(self.compiled_kernel.src.signature.items())

    @cached_property
    def block_size(self) -> Tuple[int, int, int]:
        return (
            self.compiled_kernel.metadata.num_warps
            * self.compiled_kernel.metadata.warp_size,
            1,
            1,
        )

    @cached_property
    def cubin(self) -> bytes:
        return self.compiled_kernel.asm.get("cubin")

    @cached_property
    def dynamic_shared_memory_size(self) -> int:
        return self.compiled_kernel.metadata.shared

    @cached_property
    def gpu_chip(self) -> str:
        return f"sm_{self.compiled_kernel.metadata.target.arch}"

    @cached_property
    def kernel_name(self) -> str:
        return self.compiled_kernel.metadata.name

    @cached_property
    def runtime_parameters(self) -> List[Tuple[str, str]]:
        """Get runtime parameters excluding constexpr arguments."""
        return [
            (arg_name, triton_type)
            for arg_name, triton_type in self.parameters
            if triton_type != "constexpr"
        ]

    @staticmethod
    def to_tvm_ffi_any(
        llvm_any_value: ir.Value,
        tvm_ffi_any_type: ir.Type,
    ) -> ir.Value:
        """Cast LLVM TVMFFIAny descriptor struct into !tvm_ffi.any."""
        return ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            results=[tvm_ffi_any_type],
            operands=[llvm_any_value],
        ).results[0]

    @staticmethod
    def to_llvm_dl_tensor(
        dl_tensor_value: ir.Value,
        dl_tensor_llvm_type: ir.Type,
    ) -> ir.Value:
        """Cast !dlpack.tensor into its LLVM descriptor struct type."""
        return ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            results=[dl_tensor_llvm_type],
            operands=[dl_tensor_value],
        ).results[0]

    @staticmethod
    def unbox_any(
        op_name: str,
        result_type: ir.Type,
        tvm_ffi_any_value: ir.Value,
    ) -> ir.Value:
        """Create a generic tvm_ffi unbox op and return its single result."""
        return ir.Operation.create(
            op_name,
            results=[result_type],
            operands=[tvm_ffi_any_value],
        ).results[0]

    @staticmethod
    def make_null_ptr() -> ir.Value:
        """Create a null LLVM pointer constant."""
        return llvm.mlir_zero(ir.Type.parse("!llvm.ptr"))

    def unpack_argument(
        self,
        packed_args_ptr: ir.Value,
        arg_index: int,
        triton_type: str,
    ) -> ir.Value:
        """Unpack one TVM-FFI packed argument into the expected Triton runtime value."""
        any_llvm_type = ir.Type.parse("!llvm.struct<(i32, i32, i64)>")
        tvm_ffi_any_type = ir.Type.parse("!tvm_ffi.any")
        ptr_type = ir.Type.parse("!llvm.ptr")
        i8_type = ir.IntegerType.get_signless(8)
        i64_type = ir.IntegerType.get_signless(64)

        arg_index_value = arith.constant(i64_type, arg_index)
        arg_slot_ptr = llvm.getelementptr(
            ptr_type,
            packed_args_ptr,
            [arg_index_value],
            [_LLVM_GEP_DYNAMIC_INDEX],
            any_llvm_type,
            llvm.GEPNoWrapFlags.none,
        )
        packed_any_llvm = llvm.load(any_llvm_type, arg_slot_ptr)
        packed_any = self.to_tvm_ffi_any(
            packed_any_llvm,
            tvm_ffi_any_type,
        )

        if triton_type.startswith("*"):
            dl_tensor_type = ir.Type.parse("!dlpack.tensor")
            dl_tensor_llvm_type = ir.Type.parse(
                "!llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>"
            )
            dl_tensor = self.unbox_any(
                "tvm_ffi.to_tensor",
                dl_tensor_type,
                packed_any,
            )
            dl_tensor_llvm = self.to_llvm_dl_tensor(
                dl_tensor,
                dl_tensor_llvm_type,
            )
            data_ptr = llvm.extractvalue(ptr_type, dl_tensor_llvm, [0])
            byte_offset = llvm.extractvalue(i64_type, dl_tensor_llvm, [6])
            return llvm.getelementptr(
                ptr_type,
                data_ptr,
                [byte_offset],
                [_LLVM_GEP_DYNAMIC_INDEX],
                i8_type,
                llvm.GEPNoWrapFlags.none,
            )

        if triton_type in ("fp64", "fp32", "fp16", "bf16"):
            as_f64 = self.unbox_any(
                "tvm_ffi.to_float",
                ir.F64Type.get(),
                packed_any,
            )
            if triton_type == "fp64":
                return as_f64
            float_type = {
                "fp32": ir.F32Type.get(),
                "fp16": ir.F16Type.get(),
                "bf16": ir.BF16Type.get(),
            }[triton_type]
            return arith.truncf(float_type, as_f64)

        if triton_type.startswith(("i", "u")):
            as_i64 = self.unbox_any(
                "tvm_ffi.to_int",
                i64_type,
                packed_any,
            )
            bit_width = int(triton_type[1:])
            target_int_type = ir.IntegerType.get_signless(bit_width)
            if bit_width == 64:
                return as_i64
            if bit_width < 64:
                return arith.trunci(target_int_type, as_i64)
            raise ValueError(
                f"unsupported Triton runtime integer width from TVM-FFI payload: {triton_type}"
            )

        raise ValueError(
            f"unsupported Triton runtime type for TVM-FFI argument unboxing: {triton_type}"
        )

    def build_kernel_operands(self, packed_args_ptr: ir.Value) -> List[ir.Value]:
        """Extract launch operands from TVM-FFI packed argument buffer."""
        launch_operands = [
            self.unpack_argument(packed_args_ptr, arg_index, triton_type)
            for arg_index, (_, triton_type) in enumerate(self.runtime_parameters)
        ]
        return launch_operands + [self.make_null_ptr(), self.make_null_ptr()]

    def build_gpu_launch_module(
        self,
        ctx: ir.Context,
        launch_grid: Tuple[int, ...],
    ) -> ir.Module:
        """Build an MLIR GPU launch module from compiled kernel metadata.

        Args:
            ctx: MLIR Context with all dialects registered
            launch_grid: Grid dimensions as tuple. Can be 1D (x,) or 3D (x, y, z).
                         If 1D, y and z are set to 1.

        Returns:
            ir.Module: MLIR module with gpu.binary and gpu.launch_func operations
        """
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

            # Create type constants
            index_type = ir.IndexType.get()
            i32_type = ir.IntegerType.get_signless(32)
            ptr_type = ir.Type.parse("!llvm.ptr")

            # Build gpu.object attribute with cubin binary
            cubin_mlir = "".join(f"\\{b:02X}" for b in self.cubin)
            gpu_object = ir.Attribute.parse(
                f'#gpu.object<#nvvm.target<chip = "{self.gpu_chip}">, "{cubin_mlir}">'
            )

            # Create gpu.binary operation
            module.body.append(
                ir.Operation.create(
                    "gpu.binary",
                    attributes={
                        "sym_name": ir.StringAttr.get(self.kernel_name),
                        "objects": ir.ArrayAttr.get([gpu_object]),
                        "offloadingHandler": ir.Attribute.parse("#gpu.select_object"),
                    },
                )
            )

            # Create wrapper function
            fn_name = f"__tvm_ffi_{self.kernel_name}"
            fn_type = ir.FunctionType.get(
                [ptr_type, ptr_type, i32_type, ptr_type],
                [i32_type],
            )
            fn = func.FuncOp(fn_name, fn_type)
            module.body.append(fn.operation)

            # Create entry block with function arguments
            entry = ir.Block.create_at_start(
                fn.regions[0], [ptr_type, ptr_type, i32_type, ptr_type]
            )
            with ir.InsertionPoint(entry):
                packed_args_ptr = entry.arguments[1]
                grid_dims = launch_grid + (1,) * (3 - len(launch_grid))

                # Create grid dimension constants
                grid_x_op, grid_y_op, grid_z_op = [
                    arith.constant(index_type, dim) for dim in grid_dims
                ]

                # Create block size constants
                block_x, block_y, block_z = self.block_size
                block_x_op = arith.constant(index_type, block_x)
                block_y_op = arith.constant(index_type, block_y)
                block_z_op = arith.constant(index_type, block_z)

                # Create dynamic shared memory size constant
                dynamic_smem_op = arith.constant(
                    i32_type, self.dynamic_shared_memory_size
                )

                # Create gpu.launch_func operation
                gpu.LaunchFuncOp(
                    kernel=[self.kernel_name, self.kernel_name],
                    grid_size=(grid_x_op, grid_y_op, grid_z_op),
                    block_size=(block_x_op, block_y_op, block_z_op),
                    kernel_operands=self.build_kernel_operands(packed_args_ptr),
                    dynamic_shared_memory_size=dynamic_smem_op,
                )

                # Return TVM-FFI success code.
                func.ReturnOp([arith.constant(i32_type, 0)])

        return module
