#!/bin/bash
# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT


find examples trident-core python test -name "*.py" | xargs ruff format
# clang-format corrupts some TableGen (.td) files in this repo, so only format C/C++ here.
find trident-core \( -name "*.c" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | xargs clang-format -i
(find trident-core -name CMakeLists.txt && echo CMakeLists.txt) | xargs cmake-format -i
