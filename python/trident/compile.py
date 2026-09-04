# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast, overload

from .backend import TridentGraphModule

P = ParamSpec("P")
R = TypeVar("R")


def compile(fn: Callable[P, R]) -> Callable[P, R]:
    def f(*args: P.args, **kwargs: P.kwargs) -> R:
        gm: TridentGraphModule = TridentGraphModule(fn)
        gm.compile(*args, **kwargs)
        return cast(Callable[P, R], gm)

    return f


@overload
def jit(fn: Callable[P, R], *, dynamic: bool = True) -> TridentGraphModule: ...


@overload
def jit(
    fn: None = None, *, dynamic: bool = True
) -> Callable[[Callable[P, R]], TridentGraphModule]: ...


def jit(
    fn: Callable[P, R] | None = None, *, dynamic: bool = True
) -> TridentGraphModule | Callable[[Callable[P, R]], TridentGraphModule]:
    if fn is None:

        def decorator(decorated_fn: Callable[P, R]) -> TridentGraphModule:
            return TridentGraphModule(decorated_fn, dynamic=dynamic)

        return decorator
    return TridentGraphModule(fn, dynamic=dynamic)
