from typing import Any, Callable

from .backend import LibTritonGraphModule


def compile(fn: Callable[..., Any]) -> Callable[..., Callable[..., Any]]:
    def f(*args: Any, **kwargs: Any) -> Any:
        gm: LibTritonGraphModule = LibTritonGraphModule(fn)
        gm.compile(*args, **kwargs)
        return gm.gm

    return f


def jit(fn: Callable[..., Any]) -> Callable[..., Callable[..., Any]]:
    return LibTritonGraphModule(fn)
