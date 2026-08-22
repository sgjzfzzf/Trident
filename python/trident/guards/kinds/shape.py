# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

from ..codes import ExpressionCode
from .base import Guard

if TYPE_CHECKING:
    import torch._guards


class ShapeEnvGuard(Guard):
    create_fn_name = "SHAPE_ENV"
    code_types = (ExpressionCode,)

    @classmethod
    def validate_guard(cls, guard: torch._guards.Guard) -> bool:
        return not bool(guard.name)
