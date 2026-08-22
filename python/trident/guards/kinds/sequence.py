# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from ..codes import SequenceLengthCode, TypeIdCode
from .base import Guard


class SequenceLengthGuard(Guard):
    create_fn_name = "SEQUENCE_LENGTH"
    code_types = (TypeIdCode, SequenceLengthCode)
