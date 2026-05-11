from __future__ import annotations

import unittest
from typing import Any, List, Tuple

from libtriton import guards


class _FakeEmitter(object):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: List[Tuple[Any, ...]] = []

    def emit_tensor_rank(self, tensor: str) -> str:
        self.calls.append(("rank", tensor))
        return f"rank({tensor})"

    def emit_tensor_shape_dim(self, tensor: str, dim: int) -> str:
        self.calls.append(("shape", tensor, dim))
        return f"shape({tensor},{dim})"

    def emit_tensor_dtype(self, tensor: str) -> str:
        self.calls.append(("dtype", tensor))
        return f"dtype({tensor})"

    def emit_i64(self, value: int) -> str:
        self.calls.append(("i64", value))
        return f"i64({value})"

    def emit_dtype_token(self, dtype: str) -> str:
        self.calls.append(("dtype_token", dtype))
        return f"dtype_token({dtype})"

    def emit_cmp(self, op: guards.GuardCmpOp, lhs: str, rhs: str) -> str:
        self.calls.append(("cmp", op.value, lhs, rhs))
        return f"cmp({op.value},{lhs},{rhs})"

    def emit_and(self, lhs: str, rhs: str) -> str:
        self.calls.append(("and", lhs, rhs))
        return f"and({lhs},{rhs})"


class GuardParserTest(unittest.TestCase):
    def test_parse_minimal_patterns(self) -> None:
        parser = guards.GuardParser()
        result = parser.parse(
            [
                "L['x'].ndim == 2",
                "L['x'].shape[1] >= 16",
                "L['x'].dtype == torch.float16",
            ]
        )
        self.assertEqual(3, len(result))
        self.assertIsInstance(result[0], guards.TensorRankEqExpr)
        self.assertIsInstance(result[1], guards.TensorShapeCmpExpr)
        self.assertIsInstance(result[2], guards.TensorDTypeEqExpr)

    def test_unsupported_pattern(self) -> None:
        parser = guards.GuardParser()
        result = parser.parse(["L['x'].stride()[0] == 64"])
        self.assertEqual(0, len(result))


class GuardBuilderTest(unittest.TestCase):
    def test_builder_conjunction(self) -> None:
        parser = guards.GuardParser()
        result = parser.parse(["x.ndim == 2", "x.shape[0] == 32"])
        builder = guards.GuardMlirBuilder()
        emitter = _FakeEmitter()
        condition = builder.build(result, emitter)
        self.assertIsNotNone(condition)
        self.assertTrue(str(condition).startswith("and("))


if __name__ == "__main__":
    unittest.main()
