import torch
import triton
import triton.language as tl
import libtriton


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return softmax_triton_impl(x)


@libtriton.jit
def softmax_jit(x: torch.Tensor) -> torch.Tensor:
    return softmax_triton_impl(x)


def softmax_triton_impl(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4
    y = torch.empty_like(x)
    softmax_kernel[(n_rows, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


if __name__ == "__main__":
    x = torch.randn(1823, 781, device="cuda")
    y_torch = torch.softmax(x, axis=1)
    y_triton = softmax(x)
    softmax_jit(x)
    y_jit = softmax_jit(x)
    assert torch.allclose(y_torch, y_triton), (y_torch, y_triton)
    assert torch.allclose(y_torch, y_jit), (y_torch, y_jit)
