# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .collection import Guards
from .constant import ConstantGuard  # noqa: F401
from .device import CUDADeviceGuard  # noqa: F401
from .dimension import DimensionGuard  # noqa: F401
from .dtype import DTypeGuard  # noqa: F401
from .guard import Guard
from .size import SizeGuard  # noqa: F401
from .storage_offset import StorageOffsetGuard  # noqa: F401
from .stride import StrideGuard  # noqa: F401
from .type import TensorTypeGuard  # noqa: F401

if TYPE_CHECKING:
    import torch._guards


def parse_guard(code: str) -> Optional[Guard]:
    return Guard.parse(code)


def parse_guards(guards: torch._guards.GuardsSet) -> Guards:
    return Guards([code for guard in guards for code in (guard.code_list or [])])


__all__ = [
    "parse_guard",
    "parse_guards",
]
