from __future__ import annotations

import io
import operator
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_register_expression_function() -> None:
    df = pl.DataFrame({"a": [1, None, 3]})
    ctx = pl.SQLContext(df=df)

    assert ctx.register_function("Twice", lambda value: value * 2) is ctx
    assert ctx.functions() == ["twice"]
    assert_frame_equal(
        ctx.execute("SELECT TWICE(a) AS a FROM df", eager=True),
        pl.DataFrame({"a": [2, None, 6]}),
    )

    ctx.register_function("py_add", lambda left, right: left + right)
    assert_frame_equal(
        ctx.execute("SELECT py_add(a, 2) AS a FROM df", eager=True),
        pl.DataFrame({"a": [3, None, 5]}),
    )

    empty = pl.DataFrame(schema={"a": pl.Int64})
    assert_frame_equal(
        pl.SQLContext(empty=empty)
        .register_function("twice", lambda value: value * 2)
        .execute("SELECT twice(a) AS a FROM empty", eager=True),
        empty,
    )


def test_registered_function_name_validation() -> None:
    ctx = pl.SQLContext().register_function("custom", lambda value: value)

    with pytest.raises(ValueError, match="already registered"):
        ctx.register_function("CUSTOM", lambda value: value)

    for name in [
        "SuM",
        "normalize",
        "group_concat",
        "array_agg",
        "truncate",
        "read_csv",
    ]:
        with pytest.raises(ValueError, match="built-in"):
            ctx.register_function(name, lambda value: value)

    for name in ["select", "FROM", "Group"]:
        with pytest.raises(ValueError, match="SQL keyword"):
            ctx.register_function(name, lambda value: value)

    for name in ["", "two words", "schema.function"]:
        with pytest.raises(ValueError, match="invalid SQL function name"):
            ctx.register_function(name, lambda value: value)


@pytest.mark.parametrize(
    ("query", "error"),
    [
        ("SELECT schema.twice(a) FROM df", "qualified function"),
        ("SELECT twice(a) FILTER (WHERE a > 0) FROM df", "FILTER is not supported"),
    ],
)
def test_registered_function_call_validation(query: str, error: str) -> None:
    calls = 0

    def twice(value: pl.Expr) -> pl.Expr:
        nonlocal calls
        calls += 1
        return value * 2

    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1]})).register_function("twice", twice)

    with pytest.raises(pl.exceptions.SQLInterfaceError, match=error):
        ctx.execute(query)
    assert calls == 0
    assert_frame_equal(
        ctx.execute("SELECT twice(a) AS a FROM df", eager=True),
        pl.DataFrame({"a": [2]}),
    )
    assert calls == 1


def test_registered_function_rejects_unsupported_callable_forms() -> None:
    async def async_builder(value: pl.Expr) -> pl.Expr:
        return value

    async def async_generator_builder(value: pl.Expr) -> Any:
        yield value

    def generator_builder(value: pl.Expr) -> Any:
        yield value

    class AsyncCallable:
        async def __call__(self, value: pl.Expr) -> pl.Expr:
            return value

    ctx = pl.SQLContext()
    for name, function, error in [
        ("py_async", async_builder, "async functions"),
        ("py_async_generator", async_generator_builder, "async generator functions"),
        ("py_generator", generator_builder, "generator functions"),
        ("py_async_callable", AsyncCallable(), "async functions"),
    ]:
        with pytest.raises(ValueError, match=error):
            ctx.register_function(name, function)  # type: ignore[arg-type]

    def keyword_only(value: pl.Expr, *, required: int) -> pl.Expr:
        return value + required

    with pytest.raises(ValueError, match=r"require keyword-only.*required"):
        ctx.register_function("py_keyword_only", keyword_only)
    with pytest.raises(ValueError, match="callable"):
        ctx.register_function("py_not_callable", 1)  # type: ignore[arg-type]


def test_registered_function_accepts_supported_callable_forms() -> None:
    def flexible(value: pl.Expr, scale: int = 2, *extra: pl.Expr) -> pl.Expr:
        output = value * scale
        for item in extra:
            output += item
        return output

    def positional_only(value: pl.Expr, /) -> pl.Expr:
        return value + 1

    def optional_keyword(value: pl.Expr, *, amount: int = 4) -> pl.Expr:
        return value + amount

    class BoundBuilder:
        def build(self, value: pl.Expr) -> pl.Expr:
            return value + 2

    class OpaqueBuilder:
        @property
        def __signature__(self) -> Any:
            raise ValueError

        def __call__(self, value: pl.Expr) -> pl.Expr:
            return value + 3

    df = pl.DataFrame({"a": [1]})
    ctx = pl.SQLContext(df=df)
    cases: list[tuple[str, Callable[..., pl.Expr], str, int]] = [
        ("py_flexible", flexible, "a", 2),
        ("py_variadic", flexible, "a, 3, 4", 7),
        ("py_positional", positional_only, "a", 2),
        ("py_optional_keyword", optional_keyword, "a", 5),
        ("py_bound", BoundBuilder().build, "a", 3),
        ("py_partial", partial(operator.mul, 4), "a", 4),
        ("py_builtin", operator.neg, "a", -1),
        ("py_opaque", OpaqueBuilder(), "a", 4),
    ]
    for name, function, arguments, expected in cases:
        ctx.register_function(name, function)
        assert (
            ctx.execute(f"SELECT {name}({arguments}) FROM df", eager=True).item()
            == expected
        )


def test_unregister_reregister_preserves_planned_function() -> None:
    query = "SELECT a FROM df"
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1, 2]}))
    assert query.encode() in ctx.execute(query).serialize()

    ctx.register_function("py_adjust", lambda value: value + 1)
    ctx.register_function("py_keep", lambda value: value)
    old_plan = ctx.execute("SELECT py_adjust(a) AS a FROM df")

    assert ctx.unregister_function("missing") is ctx
    assert ctx.unregister_function("PY_ADJUST") is ctx
    assert ctx.functions() == ["py_keep"]
    ctx.register_function("py_adjust", lambda value: value + 10)

    assert_frame_equal(old_plan.collect(), pl.DataFrame({"a": [2, 3]}))
    assert_frame_equal(
        ctx.execute("SELECT py_adjust(a) AS a FROM df", eager=True),
        pl.DataFrame({"a": [11, 12]}),
    )
    ctx.unregister_function("py_keep").unregister_function("py_adjust")
    assert query.encode() in ctx.execute(query).serialize()


@pytest.mark.parametrize("location", ["direct", "cast", "output_dtype", "cast_dtype"])
def test_runtime_udf_requires_return_dtype_before_execution(location: str) -> None:
    calls = 0

    def runtime(value: Any) -> Any:
        nonlocal calls
        calls += 1
        return value

    def builder(value: pl.Expr) -> pl.Expr:
        if location == "direct":
            return value.map_elements(runtime)
        if location == "cast":
            return value.map_elements(runtime).cast(pl.Int64)

        hidden = value.map_batches(runtime)
        dtype = pl.dtype_of(hidden)
        if location == "output_dtype":
            return value.map_batches(runtime, return_dtype=dtype)
        return value.cast(dtype)

    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1]}))
    ctx.register_function("py_untyped", builder)

    with pytest.raises(
        pl.exceptions.InvalidOperationError, match=r"py_untyped.*return_dtype"
    ):
        ctx.execute("SELECT py_untyped(a) FROM df")
    assert calls == 0


def test_zero_argument_expression_function() -> None:
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1, 2]}))
    ctx.register_function("py_answer", lambda: pl.lit(42, dtype=pl.Int64))
    assert_frame_equal(
        ctx.execute("SELECT py_answer() AS answer FROM df", eager=True),
        pl.DataFrame({"answer": [42, 42]}),
    )

    ctx.register_function(
        "invalid_zero_input",
        lambda: pl.map_batches([], lambda _: pl.Series([1]), return_dtype=pl.Int64),
    )
    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match=r"invalid_zero_input.*without input expressions",
    ):
        ctx.execute("SELECT invalid_zero_input() FROM df")


def test_runtime_udf_lazy_eager_streaming_and_serialization() -> None:
    builder_calls = 0

    def builder(value: pl.Expr) -> pl.Expr:
        nonlocal builder_calls
        builder_calls += 1
        return value.map_elements(
            lambda item: None if item is None else item + 1,
            return_dtype=pl.Int64,
        )

    expected = pl.DataFrame({"a": [2, None, 4]})
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1, None, 3]}))
    ctx.register_function("py_increment", builder)
    lf = ctx.execute("SELECT py_increment(a) AS a FROM df")
    assert builder_calls == 1
    assert_frame_equal(lf.collect(), expected)
    assert_frame_equal(lf.collect(engine="streaming"), expected)
    assert builder_calls == 1
    assert_frame_equal(
        pl.LazyFrame.deserialize(io.BytesIO(lf.serialize())).collect(), expected
    )
    assert_frame_equal(
        ctx.execute("SELECT py_increment(a) AS a FROM df", eager=True), expected
    )


def test_runtime_udf_schema_and_lifetime_after_unregister() -> None:
    runtime_calls = 0

    def runtime(value: int) -> int:
        nonlocal runtime_calls
        runtime_calls += 1
        return value + 1

    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1, 2]}))
    ctx.register_function(
        "py_lifetime",
        lambda value: value.map_elements(runtime, return_dtype=pl.Int64),
    )
    plan = ctx.execute("SELECT py_lifetime(a) AS a FROM df")
    assert plan.collect_schema() == pl.Schema({"a": pl.Int64})
    assert runtime_calls == 0

    ctx.unregister_function("py_lifetime")
    del ctx
    assert_frame_equal(plan.collect(), pl.DataFrame({"a": [2, 3]}))
    assert runtime_calls == 2


def test_multicolumn_batch_runtime_udf_with_literal_streaming() -> None:
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1, 2], "b": [10, 20]}))
    ctx.register_function(
        "py_batch_add",
        lambda left, right, amount: pl.map_batches(
            [left, right, amount],
            lambda columns: columns[0] + columns[1] + columns[2],
            return_dtype=pl.Int64,
        ),
    )
    plan = ctx.execute("SELECT py_batch_add(a, b, 3) AS out FROM df")
    assert_frame_equal(
        plan.collect(engine="streaming"), pl.DataFrame({"out": [14, 25]})
    )


def test_aggregate_and_window_expression_functions() -> None:
    df = pl.DataFrame({"g": ["x", "x", "y"], "v": [1, 2, 3]})
    ctx = pl.SQLContext(df=df).register_function(
        "py_native_total", lambda value: value.sum()
    )
    ctx.register_function(
        "py_runtime_total",
        lambda value: value.map_batches(
            lambda values: values.sum(),
            return_dtype=pl.Int64,
            returns_scalar=True,
        ),
    )

    assert_frame_equal(
        ctx.execute(
            "SELECT g, py_native_total(v) AS native, "
            "py_runtime_total(v) AS runtime FROM df GROUP BY g ORDER BY g",
            eager=True,
        ),
        pl.DataFrame({"g": ["x", "y"], "native": [3, 3], "runtime": [3, 3]}),
    )
    assert_frame_equal(
        ctx.execute(
            "SELECT g, py_native_total(v) OVER (PARTITION BY g) AS native, "
            "py_runtime_total(v) OVER (PARTITION BY g) AS runtime "
            "FROM df ORDER BY g, v",
            eager=True,
        ),
        pl.DataFrame(
            {
                "g": ["x", "x", "y"],
                "native": [3, 3, 3],
                "runtime": [3, 3, 3],
            }
        ),
    )


def test_function_errors_and_context_cleanup() -> None:
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1]}))
    ctx.register_function("py_identity", lambda value: value)
    with pytest.raises(pl.exceptions.ComputeError, match="py_identity"):
        ctx.execute("SELECT py_identity(a, a) FROM df")

    ctx.register_function("wrong_type", lambda value: 1)  # type: ignore[arg-type,return-value]
    with pytest.raises(pl.exceptions.ComputeError, match=r"wrong_type.*Polars Expr"):
        ctx.execute("SELECT wrong_type(a) FROM df")

    def fail(value: pl.Expr) -> pl.Expr:
        msg = "builder failed"
        raise RuntimeError(msg)

    ctx.register_function("py_fail", fail)
    with pytest.raises(pl.exceptions.ComputeError, match=r"py_fail.*builder failed"):
        ctx.execute("WITH leaked AS (SELECT * FROM df) SELECT py_fail(a) FROM leaked")
    with pytest.raises(pl.exceptions.SQLInterfaceError, match="leaked"):
        ctx.execute("SELECT * FROM leaked")
    assert_frame_equal(
        ctx.execute("SELECT * FROM df", eager=True), pl.DataFrame({"a": [1]})
    )


def test_function_callback_reentrancy_is_nonblocking() -> None:
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1]}))

    def reenter(value: pl.Expr) -> pl.Expr:
        ctx.tables()
        return value

    ctx.register_function("py_reenter", reenter)
    with pytest.raises(pl.exceptions.ComputeError, match="already in use"):
        ctx.execute("SELECT py_reenter(a) FROM df")


def test_concurrent_context_access_is_nonblocking() -> None:
    entered = threading.Event()
    release = threading.Event()
    ctx = pl.SQLContext(df=pl.DataFrame({"a": [1]}))

    def block(value: pl.Expr) -> pl.Expr:
        entered.set()
        assert release.wait(timeout=5)
        return value

    ctx.register_function("py_block", block)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ctx.execute, "SELECT py_block(a) FROM df")
        assert entered.wait(timeout=5)
        with pytest.raises(pl.exceptions.ComputeError, match="already in use"):
            ctx.tables()
        release.set()
        result = future.result()
        assert isinstance(result, pl.LazyFrame)
        assert_frame_equal(result.collect(), pl.DataFrame({"a": [1]}))
