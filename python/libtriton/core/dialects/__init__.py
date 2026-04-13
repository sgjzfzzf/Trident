import importlib

_backend = importlib.import_module("libtriton._C.libtriton_core.dialects")
__all__ = list(getattr(_backend, "__all__", []))


def __getattr__(name: str):
    if hasattr(_backend, name):
        return getattr(_backend, name)
    raise AttributeError(name)
