from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Sequence
from typing_extensions import Protocol


class GuardCmpOp(str, enum.Enum):
    EQ = "=="
    NE = "!="
    GE = ">="
    LE = "<="
    GT = ">"
    LT = "<"


@dataclasses.dataclass(frozen=True)
class GuardExpr(object):
    code: str


@dataclasses.dataclass(frozen=True)
class TensorRankEqExpr(GuardExpr):
    tensor: str
    rank: int


@dataclasses.dataclass(frozen=True)
class TensorShapeCmpExpr(GuardExpr):
    tensor: str
    dim: int
    op: GuardCmpOp
    rhs: int


@dataclasses.dataclass(frozen=True)
class TensorDTypeEqExpr(GuardExpr):
    tensor: str
    dtype: str


@dataclasses.dataclass(frozen=True)
class _RegexRule(object):
    name: str
    pattern: Pattern[str]
    builder: Callable[[str, Dict[str, str]], GuardExpr]


class GuardParser(object):
    """Regex-based parser for common tensor-related dynamo guard snippets."""

    _TENSOR_REF: str = r"(?:L\['[^']+'\]|[A-Za-z_][A-Za-z0-9_]*)"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rules: List[_RegexRule] = [
            _RegexRule(
                name="tensor-rank-eq",
                pattern=re.compile(
                    rf"^(?P<tensor>{self._TENSOR_REF})\.(?:ndim|dim\(\))\s*==\s*(?P<rank>-?\d+)\s*$"
                ),
                builder=self._build_rank_eq,
            ),
            _RegexRule(
                name="tensor-shape-cmp",
                pattern=re.compile(
                    rf"^(?P<tensor>{self._TENSOR_REF})\.(?:shape|size\(\))\s*\[\s*(?P<dim>-?\d+)\s*\]\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<rhs>-?\d+)\s*$"
                ),
                builder=self._build_shape_cmp,
            ),
            _RegexRule(
                name="tensor-dtype-eq",
                pattern=re.compile(
                    rf"^(?P<tensor>{self._TENSOR_REF})\.dtype\s*==\s*(?P<dtype>torch\.[A-Za-z0-9_]+)\s*$"
                ),
                builder=self._build_dtype_eq,
            ),
        ]

    def parse(self, codes: Iterable[str]) -> Iterable[GuardExpr]:
        return filter(
            None,
            map(lambda code: self._parse_one(code), codes),
        )

    def _parse_one(self, code: str) -> Optional[GuardExpr]:
        for rule in self._rules:
            if match := rule.pattern.match(code):
                return rule.builder(code, match.groupdict())
        return None

    @staticmethod
    def _normalize_tensor_ref(tensor_ref: str) -> str:
        if match := re.match(r"^L\['(?P<name>[^']+)'\]$", tensor_ref):
            return match.group("name")
        else:
            return tensor_ref

    @staticmethod
    def _build_rank_eq(code: str, groups: Dict[str, str]) -> GuardExpr:
        return TensorRankEqExpr(
            code=code,
            tensor=GuardParser._normalize_tensor_ref(groups["tensor"]),
            rank=int(groups["rank"]),
        )

    @staticmethod
    def _build_shape_cmp(code: str, groups: Dict[str, str]) -> GuardExpr:
        return TensorShapeCmpExpr(
            code=code,
            tensor=GuardParser._normalize_tensor_ref(groups["tensor"]),
            dim=int(groups["dim"]),
            op=GuardCmpOp(groups["op"]),
            rhs=int(groups["rhs"]),
        )

    @staticmethod
    def _build_dtype_eq(code: str, groups: Dict[str, str]) -> GuardExpr:
        return TensorDTypeEqExpr(
            code=code,
            tensor=GuardParser._normalize_tensor_ref(groups["tensor"]),
            dtype=groups["dtype"],
        )


class GuardEmitter(Protocol):
    def emit_tensor_rank(self, tensor: str) -> Any: ...

    def emit_tensor_shape_dim(self, tensor: str, dim: int) -> Any: ...

    def emit_tensor_dtype(self, tensor: str) -> Any: ...

    def emit_i64(self, value: int) -> Any: ...

    def emit_dtype_token(self, dtype: str) -> Any: ...

    def emit_cmp(self, op: GuardCmpOp, lhs: Any, rhs: Any) -> Any: ...

    def emit_and(self, lhs: Any, rhs: Any) -> Any: ...


class GuardMlirBuilder(object):
    """Builds a single conjunction condition from parsed guard expressions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def build(
        self,
        expressions: Sequence[GuardExpr],
        emitter: GuardEmitter,
    ) -> Optional[Any]:
        condition: Any = emitter.emit_i64(1)
        for expr in expressions:
            condition = emitter.emit_and(condition, self._build_one(expr, emitter))
        return condition

    @staticmethod
    def _build_one(
        expr: GuardExpr,
        emitter: GuardEmitter,
    ) -> Optional[Any]:
        if isinstance(expr, TensorRankEqExpr):
            lhs = emitter.emit_tensor_rank(expr.tensor)
            rhs = emitter.emit_i64(expr.rank)
            return emitter.emit_cmp(GuardCmpOp.EQ, lhs, rhs)
        if isinstance(expr, TensorShapeCmpExpr):
            lhs = emitter.emit_tensor_shape_dim(expr.tensor, expr.dim)
            rhs = emitter.emit_i64(expr.rhs)
            return emitter.emit_cmp(expr.op, lhs, rhs)
        if isinstance(expr, TensorDTypeEqExpr):
            lhs = emitter.emit_tensor_dtype(expr.tensor)
            rhs = emitter.emit_dtype_token(expr.dtype)
            return emitter.emit_cmp(GuardCmpOp.EQ, lhs, rhs)
        return None
