# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from trident.backend import TridentGraphModule
from trident.core import ir, register_all_dialects, register_all_passes


def _build(fn, *args, **kwargs) -> str:
    ctx = ir.Context()
    register_all_dialects(ctx)
    register_all_passes()
    return str(TridentGraphModule._build_sub_module(fn, ctx, 0, args, kwargs))


def test_default_argument_specialization() -> None:
    def fn(inp, dtype=None):
        return inp + 1

    module = _build(fn, torch.ones(2))

    assert "tvm_ffi.func @fn_0" in module
    assert "%arg1: !torch.none" in module
    assert "#tvm_ffi.ConstantGuard<type_index = 0, payload = 0>" in module
    assert "func.call @main_0(%arg0)" in module


def test_removed_middle_argument_and_keyword_order() -> None:
    def fn(a, static_middle=1, b=None):
        return a + b

    module = _build(fn, torch.ones(2), b=torch.ones(2))

    assert "%arg1: !torch.int" in module
    assert "func.call @main_0(%arg0, %arg2)" in module
