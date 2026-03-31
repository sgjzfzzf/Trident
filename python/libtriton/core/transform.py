import enum
import types

import torch
import torch.fx.passes.split_module
import torch._higher_order_ops.triton_kernel_wrap


def triton_graph_transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    transform: TritonGraphTransform = TritonGraphTransform()
    return torch.fx.passes.split_module.split_module(gm, None, transform.step)


class TritonGraphTransform(object):
    class State(enum.Enum):
        TORCH = 0
        TRITON = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnt: int = 0
        self.state: TritonGraphTransform.State = TritonGraphTransform.State.TORCH

    def step(self, node: torch.fx.Node) -> str:
        if self.state == TritonGraphTransform.State.TORCH:
            return self.step_torch(node)
        elif self.state == TritonGraphTransform.State.TRITON:
            return self.step_triton(node)
        else:
            raise ValueError(f"unknown state: {self.state}")

    def step_torch(self, node: torch.fx.Node) -> str:
        if isinstance(
            node.target,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
        ):
            self.cnt += 1
            self.state = TritonGraphTransform.State.TRITON
            return f"triton_{self.cnt}"
        else:
            return f"torch_{self.cnt}"

    def step_triton(self, node: torch.fx.Node) -> str:
        if isinstance(
            node.target,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
        ):
            self.cnt += 1
            return f"triton_{self.cnt}"
        elif (
            isinstance(
                node.target, (types.BuiltinFunctionType, types.BuiltinMethodType)
            )
            and node.name == "getitem"
        ):
            return f"triton_{self.cnt}"
        else:
            self.cnt += 1
            self.state = TritonGraphTransform.State.TORCH
            return f"torch_{self.cnt}"
