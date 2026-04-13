__all__ = ["fx_decomp_util", "fx_importer", "onnx_importer"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = __import__(f"libtriton._C.libtriton_core.extras.{name}", fromlist=[name])
    return module
