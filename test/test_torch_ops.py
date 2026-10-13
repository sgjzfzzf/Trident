# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation correctness tests against eager PyTorch.

Run with ``python -m unittest discover -s test -v``. Each case is a separate
test on CPU and CUDA (when available). Inputs are created outside the compiled
function; both its first invocation and a cached invocation with fresh values
must match eager PyTorch, including output structure, dtype, shape and device.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable

import torch
import torch.nn.functional as F
import trident


def make_inputs(kind: str, device: str, dtype: torch.dtype, seed: int) -> tuple:
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

    def indices(values: list) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.int64)

    match kind:
        case "unary":
            return (rand(3, 4),)
        case "positive":
            return (rand(3, 4).abs() + 0.5,)
        case "binary":
            return rand(3, 4), rand(3, 4)
        case "broadcast":
            return rand(3, 4), rand(4)
        case "comparison":
            x, y = rand(3, 4), rand(4)
            # Include equal, less-than and greater-than values in every case.
            x[0] = y
            x[1] = y - 1
            x[2] = y + 1
            return x, y
        case "divisor":
            return rand(3, 4), rand(4).abs() + 0.5
        case "boolean":
            return rand(3, 4) > 0, rand(3, 4) > 0
        case "boolean_unary":
            x = rand(3, 4) > 0
            x[0] = False
            x[1] = True
            return (x,)
        case "where":
            return rand(3, 4) > 0, rand(3, 4), rand(4)
        case "noncontiguous":
            return (rand(4, 3).transpose(0, 1),)
        case "singleton":
            return (rand(1, 3, 1, 4),)
        case "expand":
            return (rand(3, 1),)
        case "matrix":
            return rand(3, 5), rand(5, 4)
        case "batched_matrix":
            return rand(2, 3, 5), rand(2, 5, 4)
        case "addmm":
            return rand(3, 4), rand(3, 5), rand(5, 4)
        case "vector":
            return rand(5), rand(5)
        case "index_select":
            return rand(3, 4), indices([3, 1, 1, 0])
        case "gather":
            return rand(3, 4), indices([[3, 0], [1, 1], [0, 2]])
        case "scatter":
            return rand(3, 4), indices([[3, 0], [1, 2], [0, 2]]), rand(3, 2)
        case "embedding":
            return indices([[0, 3, 1], [4, 1, 0]]), rand(5, 4)
        case "linear":
            return rand(2, 5), rand(4, 5), rand(4)
        case "conv1d":
            return rand(2, 3, 8), rand(4, 3, 3), rand(4)
        case "conv2d":
            return rand(2, 3, 6, 6), rand(4, 3, 3, 3), rand(4)
        case "image":
            return (rand(2, 3, 6, 6),)
        case "layer_norm":
            return rand(2, 3, 4), rand(4), rand(4)
        case _:
            raise ValueError(f"Unknown input fixture: {kind}")


# Each callable contains exactly one Torch operation. Constants and keyword
# arguments exercise common overloads without introducing a multi-op model.
# (name, callable, input fixture)
CASES = (
    # Arithmetic, broadcasting and scalar overloads.
    ("add", lambda x, y: torch.add(x, y), "binary"),
    ("add_broadcast", lambda x, y: torch.add(x, y), "broadcast"),
    ("add_scalar", lambda x: torch.add(x, 1.5), "unary"),
    ("sub", lambda x, y: torch.sub(x, y, alpha=2), "broadcast"),
    ("mul", lambda x, y: torch.mul(x, y), "broadcast"),
    ("mul_scalar", lambda x: torch.mul(x, -0.5), "unary"),
    ("div", lambda x, y: torch.div(x, y), "divisor"),
    ("floor_divide", lambda x, y: torch.floor_divide(x, y), "divisor"),
    ("remainder", lambda x, y: torch.remainder(x, y), "divisor"),
    ("pow", lambda x: torch.pow(x, 2), "unary"),
    ("neg", lambda x: torch.neg(x), "unary"),
    ("abs", lambda x: torch.abs(x), "unary"),
    ("square", lambda x: torch.square(x), "unary"),
    ("sqrt", lambda x: torch.sqrt(x), "positive"),
    ("rsqrt", lambda x: torch.rsqrt(x), "positive"),
    ("reciprocal", lambda x: torch.reciprocal(x), "positive"),
    ("exp", lambda x: torch.exp(x), "unary"),
    ("log", lambda x: torch.log(x), "positive"),
    ("log1p", lambda x: torch.log1p(x), "positive"),
    ("sin", lambda x: torch.sin(x), "unary"),
    ("cos", lambda x: torch.cos(x), "unary"),
    ("tanh", lambda x: torch.tanh(x), "unary"),
    ("erf", lambda x: torch.erf(x), "unary"),
    ("floor", lambda x: torch.floor(x), "unary"),
    ("ceil", lambda x: torch.ceil(x), "unary"),
    ("round", lambda x: torch.round(x), "unary"),
    ("sign", lambda x: torch.sign(x), "unary"),
    ("clamp", lambda x: torch.clamp(x, min=-0.5, max=0.5), "unary"),
    ("maximum", lambda x, y: torch.maximum(x, y), "broadcast"),
    ("minimum", lambda x, y: torch.minimum(x, y), "broadcast"),
    # Comparisons and boolean outputs.
    ("eq", lambda x, y: torch.eq(x, y), "comparison"),
    ("ne", lambda x, y: torch.ne(x, y), "comparison"),
    ("lt", lambda x, y: torch.lt(x, y), "comparison"),
    ("le", lambda x, y: torch.le(x, y), "comparison"),
    ("gt", lambda x, y: torch.gt(x, y), "comparison"),
    ("ge", lambda x, y: torch.ge(x, y), "comparison"),
    ("logical_and", lambda x, y: torch.logical_and(x, y), "boolean"),
    ("logical_or", lambda x, y: torch.logical_or(x, y), "boolean"),
    ("logical_xor", lambda x, y: torch.logical_xor(x, y), "boolean"),
    ("logical_not", lambda x: torch.logical_not(x), "boolean_unary"),
    ("where", lambda condition, x, y: torch.where(condition, x, y), "where"),
    # Reductions, including scalar, bool and multiple-result outputs.
    ("sum", lambda x: torch.sum(x), "unary"),
    ("sum_dim", lambda x: torch.sum(x, dim=1, keepdim=True), "unary"),
    ("mean", lambda x: torch.mean(x, dim=0), "unary"),
    ("prod", lambda x: torch.prod(x, dim=1), "unary"),
    ("amax", lambda x: torch.amax(x, dim=1), "unary"),
    ("amin", lambda x: torch.amin(x, dim=0), "unary"),
    ("max", lambda x: torch.max(x, dim=1), "unary"),
    ("min", lambda x: torch.min(x, dim=0), "unary"),
    ("argmax", lambda x: torch.argmax(x, dim=1), "unary"),
    ("argmin", lambda x: torch.argmin(x, dim=0), "unary"),
    ("all", lambda x: torch.all(x, dim=1), "boolean_unary"),
    ("any", lambda x: torch.any(x, dim=1), "boolean_unary"),
    ("var", lambda x: torch.var(x, dim=1, correction=0), "unary"),
    ("std", lambda x: torch.std(x, dim=0, correction=0), "unary"),
    ("cumsum", lambda x: torch.cumsum(x, dim=1), "unary"),
    ("cumprod", lambda x: torch.cumprod(x, dim=0), "unary"),
    ("logsumexp", lambda x: torch.logsumexp(x, dim=1), "unary"),
    ("norm", lambda x: torch.linalg.vector_norm(x, dim=1), "unary"),
    # Activations and commonly used neural-network operations.
    ("relu", lambda x: F.relu(x), "unary"),
    ("sigmoid", lambda x: torch.sigmoid(x), "unary"),
    ("gelu", lambda x: F.gelu(x), "unary"),
    ("silu", lambda x: F.silu(x), "unary"),
    ("leaky_relu", lambda x: F.leaky_relu(x, negative_slope=0.1), "unary"),
    ("softmax", lambda x: F.softmax(x, dim=-1), "unary"),
    ("log_softmax", lambda x: F.log_softmax(x, dim=-1), "unary"),
    ("linear", lambda x, w, b: F.linear(x, w, b), "linear"),
    ("conv1d", lambda x, w, b: F.conv1d(x, w, b, padding=1), "conv1d"),
    ("conv2d", lambda x, w, b: F.conv2d(x, w, b, padding=1), "conv2d"),
    ("max_pool2d", lambda x: F.max_pool2d(x, kernel_size=2), "image"),
    ("avg_pool2d", lambda x: F.avg_pool2d(x, kernel_size=2), "image"),
    ("adaptive_avg_pool2d", lambda x: F.adaptive_avg_pool2d(x, (2, 2)), "image"),
    ("layer_norm", lambda x, w, b: F.layer_norm(x, (4,), w, b), "layer_norm"),
    ("embedding", lambda i, w: F.embedding(i, w), "embedding"),
    # Matrix operations.
    ("mm", lambda x, y: torch.mm(x, y), "matrix"),
    ("matmul", lambda x, y: torch.matmul(x, y), "matrix"),
    ("bmm", lambda x, y: torch.bmm(x, y), "batched_matrix"),
    ("addmm", lambda b, x, y: torch.addmm(b, x, y), "addmm"),
    ("dot", lambda x, y: torch.dot(x, y), "vector"),
    ("outer", lambda x, y: torch.outer(x, y), "vector"),
    # Shape/layout transformations and copying.
    ("reshape", lambda x: torch.reshape(x, (2, 6)), "unary"),
    ("view", lambda x: x.view(2, 6), "unary"),
    ("flatten", lambda x: torch.flatten(x), "noncontiguous"),
    ("transpose", lambda x: torch.transpose(x, 0, 1), "unary"),
    ("permute", lambda x: torch.permute(x, (1, 0)), "unary"),
    ("squeeze", lambda x: torch.squeeze(x), "singleton"),
    ("unsqueeze", lambda x: torch.unsqueeze(x, dim=1), "unary"),
    ("expand", lambda x: x.expand(3, 4), "expand"),
    ("repeat", lambda x: x.repeat(2, 1), "unary"),
    ("clone", lambda x: torch.clone(x), "noncontiguous"),
    ("contiguous", lambda x: x.contiguous(), "noncontiguous"),
    ("to_dtype", lambda x: x.to(torch.float64), "unary"),
    ("cat", lambda x, y: torch.cat((x, y), dim=1), "binary"),
    ("stack", lambda x, y: torch.stack((x, y), dim=0), "binary"),
    ("split", lambda x: torch.split(x, 2, dim=1), "unary"),
    ("chunk", lambda x: torch.chunk(x, 2, dim=1), "unary"),
    ("unbind", lambda x: torch.unbind(x, dim=0), "unary"),
    # Indexing and ordering.
    ("select", lambda x: torch.select(x, dim=1, index=2), "unary"),
    ("narrow", lambda x: torch.narrow(x, dim=1, start=1, length=2), "unary"),
    ("index_select", lambda x, i: torch.index_select(x, 1, i), "index_select"),
    ("gather", lambda x, i: torch.gather(x, 1, i), "gather"),
    ("scatter", lambda x, i, s: torch.scatter(x, 1, i, s), "scatter"),
    ("scatter_add", lambda x, i, s: torch.scatter_add(x, 1, i, s), "scatter"),
    ("sort", lambda x: torch.sort(x, dim=1, descending=True), "unary"),
    ("topk", lambda x: torch.topk(x, k=2, dim=1), "unary"),
    ("flip", lambda x: torch.flip(x, dims=(1,)), "unary"),
    ("roll", lambda x: torch.roll(x, shifts=1, dims=1), "unary"),
    # Deterministic tensor creation using the input's dtype/device.
    ("zeros_like", lambda x: torch.zeros_like(x), "unary"),
    ("ones_like", lambda x: torch.ones_like(x), "unary"),
    ("full_like", lambda x: torch.full_like(x, 2.5), "unary"),
    ("zeros", lambda x: torch.zeros((3, 4), device=x.device, dtype=x.dtype), "unary"),
    ("ones", lambda x: torch.ones((3, 4), device=x.device, dtype=x.dtype), "unary"),
    ("full", lambda x: torch.full((3, 4), -1.5, device=x.device, dtype=x.dtype), "unary"),
    ("arange", lambda x: torch.arange(0, 8, 2, device=x.device, dtype=x.dtype), "unary"),
)


class TorchOpTest(unittest.TestCase):
    """Test methods are generated so failures identify the operator and device."""

    @classmethod
    def tearDownClass(cls) -> None:
        # Matrix operations cache cuBLAS workspaces. Release them before the
        # existing tests check that no live CUDA allocations remain.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
        super().tearDownClass()


def make_test(
    name: str,
    operation: Callable,
    kind: str,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> Callable:
    # Trident uses the function name as a JIT symbol, so avoid '<lambda>'.
    operation.__name__ = name

    def test(self: TorchOpTest) -> None:
        compiled = trident.jit(operation, dynamic=False)
        for seed in (17, 29):
            with self.subTest(seed=seed):
                args = make_inputs(kind, device, dtype, seed)
                expected = operation(*args)
                actual = compiled(*args)
                torch.testing.assert_close(actual, expected)

    test.__doc__ = f"{name} matches eager PyTorch on {device} ({dtype})."
    if device == "cuda":
        test = unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")(test)
    return test


# A small low-precision matrix for commonly used GPU operations, in addition
# to the complete CPU/CUDA float32 suite above.
LOW_PRECISION_OPS = {
    "add", "mul", "div", "relu", "gelu", "silu", "softmax", "layer_norm",
    "mm", "bmm", "linear", "conv2d",
}

for _name, _operation, _kind in CASES:
    for _device in ("cpu", "cuda"):
        setattr(
            TorchOpTest,
            f"test_{_name}_{_device}",
            make_test(_name, _operation, _kind, _device),
        )
    if _name in LOW_PRECISION_OPS:
        for _dtype in (torch.float16, torch.bfloat16):
            setattr(
                TorchOpTest,
                f"test_{_name}_cuda_{str(_dtype).removeprefix('torch.')}",
                make_test(_name, _operation, _kind, "cuda", _dtype),
            )


if __name__ == "__main__":
    unittest.main()
