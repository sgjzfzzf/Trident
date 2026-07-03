# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Final

import torch
import triton

from trident._C.trident_core import ir
from trident._C.trident_core.dialects import (
    arith,
    func,
    gpu,
    torch as torch_d,
    torchext,
)
from trident._C.trident_core.extras.fx_importer import FxImporter, GraphNodeImporter


class TridentGraphNodeImporter(GraphNodeImporter):
    """GraphNodeImporter subclass that handles triton higher-order ops natively."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _runtime_parameters(
        kernel: triton.compiler.CompiledKernel,
    ) -> List[Tuple[str, str]]:
        return [
            (arg_name, triton_type)
            for arg_name, triton_type in kernel.src.signature.items()
            if triton_type != "constexpr"
        ]

    @staticmethod
    def _add_gpu_binary(
        loc: Any,
        module: ir.Module,
        binary_name: str,
        kernel: triton.compiler.CompiledKernel,
    ) -> None:
        module_op = module.operation
        module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
        if all(
            not isinstance(op, gpu.BinaryOp) or op.sym_name != binary_name
            for op in module.body.operations
        ):
            cubin = kernel.asm["cubin"]
            cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
            gpu_object = ir.Attribute.parse(
                '#gpu.object<#nvvm.target<chip = "sm_{arch}">, "{cubin}">'.format(
                    arch=kernel.metadata.target.arch,
                    cubin=cubin_mlir,
                )
            )
            with ir.InsertionPoint(module.body):
                gpu.binary(
                    ir.StringAttr.get(binary_name),
                    ir.ArrayAttr.get([gpu_object]),
                    offloading_handler=ir.Attribute.parse("#gpu.select_object"),
                    loc=loc,
                )

    def _materialize_runtime_argument(
        self,
        loc: Any,
        name: str,
        triton_type: str,
        imported_values: Dict[str, ir.Value],
        constant_args: Dict[str, Any],
    ) -> ir.Value:
        if value := imported_values.get(name):
            return value
        elif value := constant_args.get(name):
            with loc:
                if triton_type.startswith("*") and value is None:
                    return self._make_null_ptr()
                elif triton_type.startswith(("i", "u")):
                    return torch_d.ConstantIntOp(int(value))
                elif triton_type in ("fp16", "bf16", "fp32", "fp64"):
                    return torch_d.ConstantFloatOp(float(value))
                else:
                    assert False, f"unsupported constant argument type: {triton_type}"
        else:
            assert False, f"missing runtime argument for {name} of type {triton_type}"

    def _emit_trident_kernel_launch(
        self,
        loc: Any,
        kernel_attr: ir.Attribute,
        grid: Tuple[int, int, int],
        block_size_x: int,
        operands: List[ir.Value],
        dynamic_shared_memory_size: int,
    ) -> None:
        index_type = ir.IndexType.get()
        i32_type = ir.IntegerType.get_signless(32)
        grid_x, grid_y, grid_z = grid
        torchext.TridentKernelLaunchOp(
            kernel_attr,
            arith.constant(index_type, grid_x, loc=loc),
            arith.constant(index_type, grid_y, loc=loc),
            arith.constant(index_type, grid_z, loc=loc),
            arith.constant(index_type, block_size_x, loc=loc),
            arith.constant(index_type, 1, loc=loc),
            arith.constant(index_type, 1, loc=loc),
            operands,
            dynamicSharedMemorySize=arith.constant(
                i32_type, dynamic_shared_memory_size, loc=loc
            ),
            loc=loc,
        )

    def _import_hop_triton_kernel_wrapper_mutation(
        self,
        loc: Any,
        node: torch.fx.Node,
        hop: Any,
    ) -> None:
        knodes: Dict[str, Any] = node.kwargs["kwargs"]
        ktensors: Dict[str, torch.Tensor] = {
            name: triton.MockTensor(dtype=value.meta["val"].dtype)
            for name, value in knodes.items()
        }
        kvalues: Dict[str, ir.Value] = {
            name: self._import_argument(loc, value) for name, value in knodes.items()
        }
        output_names: List[str] = node.kwargs.get("tensors_to_clone", [])
        constant_args_idx: Final[int] = node.kwargs["constant_args_idx"]
        constant_args: Dict[str, Any] = (
            torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_constant_args(
                constant_args_idx
            )
        )
        kernel_idx: Final[int] = node.kwargs["kernel_idx"]
        function: triton.KernelInterface = (
            torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
                kernel_idx
            )
        )
        kernel: triton.compiler.CompiledKernel = self.get_compiled_kernel(
            function,
            [],
            ktensors | constant_args,
        )
        assert kernel is not None, (
            f"failed to get compiled Triton kernel for {node.name}"
        )
        runtime_parameters: List[Tuple[str, str]] = self._runtime_parameters(kernel)
        call_arguments: Dict[str, ir.Value] = {
            name: self._materialize_runtime_argument(
                loc,
                name,
                triton_type,
                kvalues,
                constant_args,
            )
            for name, triton_type in runtime_parameters
        }
        operands: List[ir.Value] = [
            call_arguments[name] for name, _ in runtime_parameters
        ]
        raw_grid = node.kwargs["grid"]
        # With @triton.autotune, the grid is a list of grids (one per config).
        # Without autotune, it's a single 3-tuple.
        if (
            isinstance(raw_grid, (list, tuple))
            and len(raw_grid) > 0
            and isinstance(raw_grid[0], (list, tuple))
        ):
            # List of grids from autotune — pick the grid matching the best config.
            if hasattr(function, "best_config"):
                best_kwargs = function.best_config.all_kwargs()
                M_val = constant_args.get("M", 0)
                N_val = constant_args.get("N", 0)
                block_m = best_kwargs.get("BLOCK_SIZE_M", 0)
                block_n = best_kwargs.get("BLOCK_SIZE_N", 0)
                if M_val and N_val and block_m and block_n:
                    import math

                    grid_x = math.ceil(int(M_val) / int(block_m)) * math.ceil(
                        int(N_val) / int(block_n)
                    )
                    grid = (grid_x, 1, 1)
                else:
                    grid = tuple(raw_grid[0])
            else:
                grid = tuple(raw_grid[0])
        else:
            grid = tuple(raw_grid)
        binary_name: Final[str] = f"_{node.name}_{random.randint(0, 1 << 32)}"
        self._add_gpu_binary(loc, self.fx_importer.module, binary_name, kernel)
        self._emit_trident_kernel_launch(
            loc,
            ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
            grid,
            kernel.metadata.num_warps * kernel.metadata.warp_size,
            operands,
            kernel.metadata.shared,
        )

        self._multi_result_nodes.add(node)

        for output_name in output_names:
            if output_name in call_arguments:
                self.bind_node_value(node, call_arguments[output_name], output_name)

    @staticmethod
    def get_compiled_kernel(
        function: triton.KernelInterface, args: List[Any], kwargs: Dict[str, Any]
    ) -> Optional[triton.KernelInterface]:
        device = triton.runtime.driver.active.get_current_device()
        best_config: Dict[str, Any] = (
            function.best_config.all_kwargs()
            if hasattr(function, "best_config")
            else {}
        )
        while not isinstance(function, triton.JITFunction):
            function = function.fn
        kernel_cache, kernel_key_cache, _, _, binder = function.device_caches[device]
        _, specialization, options = binder(
            *args,
            **kwargs,
            **best_config,
            debug=kwargs.get("debug", function.debug) or triton.knobs.runtime.debug,
            instrumentation_mode=triton.knobs.compilation.instrumentation_mode,
        )
        key: str = triton.runtime.jit.compute_cache_key(
            kernel_key_cache, specialization, options
        )
        return kernel_cache.get(key)


class TridentFxImporter(FxImporter):
    """FxImporter subclass that uses TridentGraphNodeImporter for triton op support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def import_stateless_graph(
        self,
        g: torch.fx.Graph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Any:
        """Override to inject TridentGraphNodeImporter."""
        ftype, loc = self._graph_to_function_meta(g)
        with loc:
            func_op = func.FuncOp(
                func_name,
                ftype,
                ip=self._m_ip,
                visibility=func_visibility,
            )
            entry_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
        node_importer = TridentGraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
        )
        node_importer.import_nodes(
            g.nodes,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )
        self.symbol_table.insert(func_op)
        return func_op
