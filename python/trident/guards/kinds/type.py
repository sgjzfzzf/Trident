# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from ..codes import TypeIdCode
from .base import Guard


class TypeMatchGuard(Guard):
    create_fn_name = "TYPE_MATCH"
    code_types = (TypeIdCode,)
