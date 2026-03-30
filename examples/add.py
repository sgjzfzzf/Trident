from typing import Any, List

from libtriton._C.torch_mlir import fx
import torch
from torch._dynamo.backends.common import aot_autograd
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def customized_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> torch.fx.GraphModule:
    gm.print_readable()
    m = fx.stateless_fx_import(gm)
    print(m)
    return gm.forward


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.compile(backend=aot_autograd(fw_compiler=customized_backend))
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output: torch.Tensor = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements: int = output.numel()
    BLOCK_SIZE: int = 1024
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)](
        x, y, output, n_elements, BLOCK_SIZE
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
