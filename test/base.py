# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Base test class for ATen op end-to-end tests.

Subclasses only need to implement ``op_name`` (e.g. ``"empty_like"``) and the
base class handles parsing, compilation, JIT, and ``tvm_ffi.Function`` wrapping.
"""

from __future__ import annotations

from abc import abstractmethod
import pathlib
import unittest
from typing import Dict, Final, List

import tvm_ffi

from trident._C.trident_core import (
    capi_utils,
    execution_engine,
    ir,
    passmanager,
)
from trident._C.trident_core._mlir_libs._tridentCore import (
    register_all_dialects,
    register_all_passes,
)


class AtenOpTest(unittest.TestCase):
    """Base class for ATen op end-to-end tests.

    Subclasses must implement the ``op_name`` classmethod::

        class MyOpTest(AtenOpTest):
            @classmethod
            def op_name(cls) -> str:
                return "my_op"
    """

    @classmethod
    @abstractmethod
    def op_name(cls) -> str: ...

    def setUp(self) -> None:
        if self.__class__ is AtenOpTest:
            raise self.skipTest("AtenOpTest is an abstract base class")
        register_all_passes()

        op: str = self.op_name()
        mlir_path: pathlib.Path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "trident-core"
            / "test"
            / "Conversion"
            / "Pipeline"
            / f"{op}.mlir"
        )

        ctx: ir.Context = ir.Context()
        register_all_dialects(ctx)
        module: ir.Module = ir.Module.parse(mlir_path.read_text(), ctx)

        with ctx:
            pm: passmanager.PassManager = passmanager.PassManager.parse(
                "builtin.module(torch-to-llvm-pipeline)",
            )
            pm.run(module.operation)

        shared_libs: Final[List[str]] = capi_utils.find_runtime_libraries()
        self._engine: execution_engine.ExecutionEngine = (
            execution_engine.ExecutionEngine(module, shared_libs=shared_libs)
        )
        self._ffi_funcs: Dict[str, tvm_ffi.Function] = {}

    def get_ffi_func(self, func_name: str) -> tvm_ffi.Function:
        """Return a wrapped ``tvm_ffi.Function`` by exported function name."""

        if func_name not in self._ffi_funcs:
            func_ptr: int = self._engine.raw_lookup(f"__tvm_ffi_{func_name}")
            self.assertNotEqual(
                func_ptr,
                0,
                msg=f"raw_lookup returned 0 for __tvm_ffi_{func_name}",
            )

            ffi_func: tvm_ffi.Function = (
                tvm_ffi.Function.__from_mlir_packed_safe_call__(
                    func_ptr,
                    keep_alive_object=self._engine,
                )
            )
            self._ffi_funcs[func_name] = ffi_func
        return self._ffi_funcs[func_name]
