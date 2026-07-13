#!/bin/bash
# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT


find examples core ffi python test -name "*.py" | xargs ruff format
# clang-format corrupts some TableGen (.td) files in this repo, so only format C/C++ here.
find core ffi \( -name "*.c" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | xargs clang-format -i
(find core ffi -name CMakeLists.txt && echo CMakeLists.txt) | xargs cmake-format -i
