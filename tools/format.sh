#!/bin/bash

find examples libtriton-core python test -name "*.py" | xargs ruff format
find libtriton-core -name "*.c" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" -o -name "*.td" | xargs clang-format -i
