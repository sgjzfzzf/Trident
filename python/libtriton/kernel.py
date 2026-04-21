from __future__ import annotations

from functools import cached_property
from typing import Any, Final, List, Tuple, TYPE_CHECKING

import libtriton._C.libtriton_core as mlir_core
from libtriton._C.libtriton_core.dialects import arith, func, gpu

if TYPE_CHECKING:
    import triton


class KernelBuilder:
    """Builder for constructing MLIR GPU launch modules from Triton compiled kernels."""

    def __init__(self, compiled_kernel: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_kernel: Final[triton.compiler.CompiledKernel] = compiled_kernel

    @staticmethod
    def triton_type_to_mlir_type(triton_type: str) -> mlir_core.ir.Type:
        """Map Triton signature type string to MLIR type."""
        if triton_type.startswith("*"):
            return mlir_core.ir.Type.parse("!llvm.ptr")
        elif triton_type == "bf16":
            return mlir_core.ir.BF16Type.get()
        elif triton_type == "fp16":
            return mlir_core.ir.F16Type.get()
        elif triton_type == "fp32":
            return mlir_core.ir.F32Type.get()
        elif triton_type == "fp64":
            return mlir_core.ir.F64Type.get()
        elif triton_type.startswith(("i", "u")):
            return mlir_core.ir.IntegerType.get_signless(int(triton_type[1:]))
        else:
            raise ValueError(f"unsupported Triton runtime type: {triton_type}")

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

    @cached_property
    def get_argument_types(self) -> List[mlir_core.ir.Type]:
        """Get MLIR types for runtime kernel arguments."""
        return [
            self.triton_type_to_mlir_type(triton_type)
            for _, triton_type in self.runtime_parameters
        ]

    @staticmethod
    def get_launch_operands(entry: mlir_core.ir.Block) -> List[mlir_core.ir.Value]:
        """Extract launch operands from entry block arguments."""
        return list(entry.arguments)

    def build_gpu_launch_module(
        self,
        ctx: mlir_core.ir.Context,
        launch_grid: Tuple[int, ...],
        fn_name: str = "launch_kernel",
    ) -> mlir_core.ir.Module:
        """Build an MLIR GPU launch module from compiled kernel metadata.

        Args:
            ctx: MLIR Context with all dialects registered
            launch_grid: Grid dimensions as tuple. Can be 1D (x,) or 3D (x, y, z).
                         If 1D, y and z are set to 1.
            fn_name: Name of the launch wrapper function (default: "launch_kernel")

        Returns:
            ir.Module: MLIR module with gpu.binary and gpu.launch_func operations
        """
        with ctx, mlir_core.ir.Location.unknown():
            module = mlir_core.ir.Module.create()
            module.operation.attributes["gpu.container_module"] = (
                mlir_core.ir.UnitAttr.get()
            )

            # Create type constants
            index_type = mlir_core.ir.IndexType.get()
            i32_type = mlir_core.ir.IntegerType.get_signless(32)

            # Get kernel operand types from signature
            kernel_operand_types = self.get_argument_types

            # Build gpu.object attribute with cubin binary
            cubin_mlir = "".join(f"\\{b:02X}" for b in self.cubin)
            gpu_object = mlir_core.ir.Attribute.parse(
                f'#gpu.object<#nvvm.target<chip = "{self.gpu_chip}">, "{cubin_mlir}">'
            )

            # Create gpu.binary operation
            module.body.append(
                mlir_core.ir.Operation.create(
                    "gpu.binary",
                    attributes={
                        "sym_name": mlir_core.ir.StringAttr.get(self.kernel_name),
                        "objects": mlir_core.ir.ArrayAttr.get([gpu_object]),
                        "offloadingHandler": mlir_core.ir.Attribute.parse(
                            "#gpu.select_object"
                        ),
                    },
                )
            )

            # Create wrapper function
            fn_type = mlir_core.ir.FunctionType.get(kernel_operand_types, [])
            fn = func.FuncOp(fn_name, fn_type)
            module.body.append(fn.operation)

            # Create entry block with function arguments
            entry = mlir_core.ir.Block.create_at_start(
                fn.regions[0], kernel_operand_types
            )
            with mlir_core.ir.InsertionPoint(entry):
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
                    kernel_operands=self.get_launch_operands(entry),
                    dynamic_shared_memory_size=dynamic_smem_op,
                )

                # Return from wrapper function
                func.ReturnOp([])

        return module
