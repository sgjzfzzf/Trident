from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Guard, Guards
from .check import CheckGuard  # noqa: F401
from .dimension import DimensionGuard  # noqa: F401
from .device import DeviceGuard  # noqa: F401
from .dtype import DTypeGuard  # noqa: F401
from .size import SizeGuard  # noqa: F401
from .storage_offset import StorageOffsetGuard  # noqa: F401
from .stride import StrideGuard  # noqa: F401
from .type import TensorTypeGuard  # noqa: F401
from .uncheck import UnCheckGuard  # noqa: F401

if TYPE_CHECKING:
    import torch._guards


def parse_guard(code: str) -> Guard:
    return Guard.parse(code)


def parse_guards(guards: torch._guards.GuardsSet) -> Guards:
    return Guards([code for guard in guards for code in (guard.code_list or [])])


__all__ = [
    "parse_guard",
    "parse_guards",
]
