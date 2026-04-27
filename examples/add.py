from libtriton import triton_graph_backend
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


@torch.compile(backend=triton_graph_backend)
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
    for _ in range(5):
        # warmup
        add(x, y)
    output_triton = add(x, y)
    # TODO: a known issue is that `output_triton` will show its device as `cpu` when it's actually on `cuda`. This is because tvm_ffi assign the wrong device type to it.
    torch.testing.assert_close(output_triton.cpu(), output_torch.cpu())
