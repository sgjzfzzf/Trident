# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any

from .backend import TridentGraphModule


def compile(fn: Callable[..., Any]) -> Callable[..., Callable[..., Any]]:
    def f(*args: Any, **kwargs: Any) -> Any:
        gm: TridentGraphModule = TridentGraphModule(fn)
        gm.compile(*args, **kwargs)
        return gm.gm

    return f


def jit(fn: Callable[..., Any]) -> TridentGraphModule:
    return TridentGraphModule(fn)
