from typing_extensions import Any, Dict, List


from libtriton._C.torch_mlir import fx, ir
import torch

from .transform import triton_graph_transform


def triton_graph_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> torch.fx.GraphModule:
    modules: Dict[str, ir.Module] = {}
    gm: torch.fx.GraphModule = triton_graph_transform(gm)
    for name, child in gm.named_children():
        if name.startswith("submod_torch_"):
            modules[name] = fx.stateless_fx_import(child, model_name=name)
        elif name.startswith("submod_triton_"):
            ...
        else:
            raise ValueError(f"unknown submodule type: {name}")
    return gm
