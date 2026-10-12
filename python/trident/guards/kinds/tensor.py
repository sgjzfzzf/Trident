# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from ..codes import (
    DynamoAttributeAbsentCode,
    RequiresGradCode,
    TensorDTypeCode,
    TensorDeviceCode,
    TensorRankCode,
    TypeIdCode,
)
from .base import Guard


class TensorMatchGuard(Guard):
    create_fn_name = "TENSOR_MATCH"
    code_types = (
        DynamoAttributeAbsentCode,
        RequiresGradCode,
        TensorDTypeCode,
        TensorDeviceCode,
        TensorRankCode,
        TypeIdCode,
    )
