from __future__ import annotations

import pkgutil
from typing import Any, Callable

from .error import GuardMatchException

__path__ = pkgutil.extend_path(__path__, __name__)


def compile(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
    from .compile import compile as _compile

    return _compile(fn, *args, **kwargs)


__all__ = [
    "compile",
    "GuardMatchException",
]
