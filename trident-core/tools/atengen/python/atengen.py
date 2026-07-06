#===----------------------------------------------------------------------===#
#
# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===#

"""
atengen — Schema-driven C++ code generator for ATen operator wrappers.

This module reads PyTorch operator schemas and emits C++ wrapper functions
that use atengen::Function<Ret, Args<...>>::call() to:
  1. Decode each TVMFFIAny argument via atengen::buildValue<T>().
  2. Push IValues onto a torch::jit::Stack.
  3. Call c10::OperatorHandle::callBoxed() with the stack.
  4. Pop results and encode back via atengen::resolveValue<T>().

Usage:
    python atengen.py                     # all aten operators → stdout
    python atengen.py --op add.Tensor mm  # specific operators → stdout
    python atengen.py --out wrappers.cc   # all aten operators → file
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import os
from typing import Final, Iterator, List, Tuple

import jinja2
import torch

_TEMPLATE_DIR: Final[Path] = Path(__file__).parent / "templates"


# ===========================================================================
# JitType -> C++ template argument helpers
# ===========================================================================


def _type_kind_expr(jit_type: torch.JitType) -> str:
    """Convert a JitType to C++ template argument string.

    Uses jit_type.kind() for the enum name and jit_type.containedTypes()
    for compound type arguments. If there are contained types, they are
    joined with commas inside SubTypes<...>.

    Examples:
        TensorType                     → "atengen::Kind<c10::TypeKind::TensorType>"
        OptionalType(IntType)          → "atengen::Contain<atengen::Kind<c10::TypeKind::OptionalType>, atengen::SubTypes<atengen::Kind<c10::TypeKind::IntType>>>"
        ListType(IntType)              → "atengen::Contain<atengen::Kind<c10::TypeKind::ListType>, atengen::SubTypes<atengen::Kind<c10::TypeKind::IntType>>>"
        DictType(IntType, TensorType)  → "atengen::Contain<atengen::Kind<c10::TypeKind::DictType>, atengen::SubTypes<atengen::Kind<c10::TypeKind::IntType>, atengen::Kind<c10::TypeKind::TensorType>>>"
    """

    if contained := jit_type.containedTypes():
        inner: Final[str] = ", ".join(_type_kind_expr(t) for t in contained)
        return (
            f"atengen::Contain<atengen::Kind<c10::TypeKind::{jit_type.kind()}>, "
            f"atengen::SubTypes<{inner}>>"
        )
    else:
        return f"atengen::Kind<c10::TypeKind::{jit_type.kind()}>"


# ===========================================================================
# Schema -> Tuple[str, str, str]
# ===========================================================================


def schema_to_descriptor(schema: torch.FunctionSchema) -> Tuple[str, str, str]:
    name: Final[str] = schema.name.removeprefix("aten::")

    # --- Build ret_expr ---
    if len(schema.returns) == 0:
        ret_expr: Final[str] = "atengen::Kind<c10::TypeKind::NoneType>"
    elif len(schema.returns) == 1:
        [ret] = schema.returns
        ret_expr: Final[str] = _type_kind_expr(ret.type)
    else:
        inner: Final[str] = ", ".join(_type_kind_expr(r.type) for r in schema.returns)
        ret_expr: Final[str] = (
            f"atengen::Contain<atengen::Kind<c10::TypeKind::TupleType>, "
            f"atengen::SubTypes<{inner}>>"
        )

    # --- Build args_expr ---
    args_expr: Final[str] = (
        f"atengen::Args<{', '.join(_type_kind_expr(arg.type) for arg in schema.arguments)}>"
    )

    return (
        name,
        schema.overload_name,
        f"atengen::Function<{ret_expr}, {args_expr}>::call",
    )


# ===========================================================================
# Code generation
# ===========================================================================


def generate_source(operators: List[Tuple[str, str, str]]) -> str:
    env: jinja2.Environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
        keep_trailing_newline=True,
    )
    template: jinja2.Template = env.get_template("aten.cc.j2")
    return template.render(operators=operators)


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="atengen",
        description="Generate C++ wrapper code from ATen operator schemas.",
    )
    parser.add_argument(
        "--op",
        nargs="+",
        default=None,
        help="Operator names to generate wrappers for (e.g., --op add.Tensor mm). "
        "If not specified, all aten operators are generated.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path for generated source code. "
        "If not specified, output goes to stdout.",
    )
    args = parser.parse_args()

    if args.op is None:
        schemas: Iterator[torch.FunctionSchema] = filter(
            lambda s: s.name.startswith("aten::"), torch._C._jit_get_all_schemas()
        )
    else:
        schemas: Iterator[torch.FunctionSchema] = itertools.starmap(
            lambda base, overload: torch._C._get_schema(f"aten::{base}", overload),
            (name.split(".", 1) if "." in name else (name, "") for name in args.op),
        )

    operators: List[Tuple[str, str, str]] = [schema_to_descriptor(s) for s in schemas]
    source_content: Final[str] = generate_source(operators)

    if args.out:
        source_dir = os.path.dirname(args.out)
        if source_dir:
            os.makedirs(source_dir, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(source_content)
    else:
        print(source_content)


if __name__ == "__main__":
    main()
