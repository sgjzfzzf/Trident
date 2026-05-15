from typing import Any, Callable

from .backend import TritonGraphModule


def compile(fn: Callable[..., Any]) -> Callable[..., Callable[..., Any]]:

    def f(*args: Any, **kwargs: Any) -> Any:
        gm: TritonGraphModule = TritonGraphModule(fn)
        gm.compile(*args, **kwargs)
        return gm.executor

    return f


def jit(fn: Callable[..., Any]) -> Callable[..., Callable[..., Any]]:
    return TritonGraphModule(fn)
