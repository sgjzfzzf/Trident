import libtriton
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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


@libtriton.jit
def add_jit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return add_impl(x, y)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return add_impl(x, y)


def add_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    for ex in range(12, 14):
        size = 2**ex
        x = torch.rand(size, device=DEVICE)
        y = torch.rand(size, device=DEVICE)
        output_torch = x + y
        output_triton = add(x, y)
        add_jit(x, y)  # warmup
        output_jit = add_jit(x, y)
        torch.testing.assert_close(output_triton, output_torch)
        torch.testing.assert_close(output_jit, output_torch)
