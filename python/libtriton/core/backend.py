from typing import Any

import torch

from libtriton._C.libtriton_core import fx, ir
from .transform import triton_graph_transform


def triton_graph_backend(
    gm: torch.fx.GraphModule, example_inputs: list[Any]
) -> torch.fx.GraphModule:
    _ = example_inputs
    modules: dict[str, ir.Module] = {}
    gm = triton_graph_transform(gm)
    for name, child in gm.named_children():
        if name.startswith("submod_torch_"):
            modules[name] = fx.stateless_fx_import(child, model_name=name)
        elif name.startswith("submod_triton_"):
            ...
        else:
            raise ValueError(f"unknown submodule type: {name}")
    return gm
