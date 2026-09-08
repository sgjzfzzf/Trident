# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import torch

from .. import ir
from .._mlir_libs._trident import DTypeType  # noqa: F401
from ._torchext_ops_gen import *


def dtype(value: torch.dtype, *, context: ir.Context | None = None) -> ir.Attribute:
    """Return the TorchExt attribute for a Torch dtype value."""

    attr_name = f"{value}".removeprefix("torch.")
    return ir.Attribute.parse(
        f"#torchext.{attr_name}",
        context=context,
    )
