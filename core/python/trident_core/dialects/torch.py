# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from .._mlir_libs._torchMlir import register_dialect  # noqa: F401
from .._mlir_libs._trident import (
    TorchAnyType,  # noqa: F401
    TorchBoolType,  # noqa: F401
    TorchDeviceType,  # noqa: F401
    TorchFloatType,  # noqa: F401
    TorchGeneratorType,  # noqa: F401
    TorchIntType,  # noqa: F401
    TorchListType,  # noqa: F401
    TorchNonValueTensorType,  # noqa: F401
    TorchTupleType,  # noqa: F401
    TorchValueTensorType,  # noqa: F401
)
from ._torch_ops_gen import *
