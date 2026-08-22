# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

from ..codes.base import GuardCode
from ..local import Local
from .base import Guard

if TYPE_CHECKING:
    import torch._guards


class IgnoredGuard(Guard):
    """Base implementation for ambient guards with no wrapper IR check."""

    @classmethod
    def parse_source(
        cls,
        guard: torch._guards.Guard,
    ) -> Local | None:
        return None

    @classmethod
    def parse_codes(
        cls,
        texts: tuple[str, ...],
        source: Local | None,
    ) -> tuple[GuardCode, ...]:
        return ()


class AutogradSavedTensorsHooksGuard(IgnoredGuard):
    create_fn_name = "AUTOGRAD_SAVED_TENSORS_HOOKS"


class DefaultDeviceGuard(IgnoredGuard):
    create_fn_name = "DEFAULT_DEVICE"


class DeterministicAlgorithmsGuard(IgnoredGuard):
    create_fn_name = "DETERMINISTIC_ALGORITHMS"


class GlobalStateGuard(IgnoredGuard):
    create_fn_name = "GLOBAL_STATE"


class GradModeGuard(IgnoredGuard):
    create_fn_name = "GRAD_MODE"


class TorchFunctionStateGuard(IgnoredGuard):
    create_fn_name = "TORCH_FUNCTION_STATE"
