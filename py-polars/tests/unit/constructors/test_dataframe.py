from __future__ import annotations

import dataclasses
import enum
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars._utils.construction.dataframe import (
    _resolved_sequence_handlers,
    _sequence_of_dataclasses_to_pydf,
    _sequence_to_pydf_dispatcher,
)
from polars.exceptions import (
    ComputeError,
    DataOrientationWarning,
    InvalidOperationError,
)
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars._typing import SchemaDict


def test_df_mixed_dtypes_string() -> None:
    data = {"x": [["abc", 12, 34.5]], "y": [1]}

    with pytest.raises(TypeError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {"x": pl.List(pl.String), "y": pl.Int64}
    assert df.rows() == [(["abc", "12", "34.5"], 1)]


def test_df_mixed_dtypes_object() -> None:
    data = {"x": [[b"abc", 12, 34.5]], "y": [1]}

    with pytest.raises(TypeError):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {"x": pl.Object, "y": pl.Int64}
    assert df.rows() == [([b"abc", 12, 34.5], 1)]


def test_df_object() -> None:
    class Foo:
        def __init__(self, value: int) -> None:
            self._value = value

        def __eq__(self, other: object) -> bool:
            return issubclass(other.__class__, self.__class__) and (
                self._value == other._value  # type: ignore[attr-defined]
            )

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self._value})"

    df = pl.DataFrame({"a": [Foo(1), Foo(2)]})
    assert df["a"].dtype.is_object()
    assert df.rows() == [(Foo(1),), (Foo(2),)]


def test_df_init_from_generator_dict_view() -> None:
    d = {0: "x", 1: "y", 2: "z"}
    data = {
        "keys": d.keys(),
        "vals": d.values(),
        "items": d.items(),
    }
    with pytest.raises(TypeError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {
        "keys": pl.Int64,
        "vals": pl.String,
        "items": pl.List(pl.String),
    }
    assert df.to_dict(as_series=False) == {
        "keys": [0, 1, 2],
        "vals": ["x", "y", "z"],
        "items": [["0", "x"], ["1", "y"], ["2", "z"]],
    }


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="reversed dict views not supported before Python 3.11",
)
def test_df_init_from_generator_reversed_dict_view() -> None:
    d = {0: "x", 1: "y", 2: "z"}
    data = {
        "rev_keys": reversed(d.keys()),
        "rev_vals": reversed(d.values()),
        "rev_items": reversed(d.items()),
    }
    df = pl.DataFrame(data, schema_overrides={"rev_items": pl.Object})

    assert df.schema == {
        "rev_keys": pl.Int64,
        "rev_vals": pl.String,
        "rev_items": pl.Object,
    }
    assert df.to_dict(as_series=False) == {
        "rev_keys": [2, 1, 0],
        "rev_vals": ["z", "y", "x"],
        "rev_items": [(2, "z"), (1, "y"), (0, "x")],
    }


def test_df_init_strict() -> None:
    data = {"a": [1, 2, 3.0]}
    schema = {"a": pl.Int8}
    with pytest.raises(TypeError):
        pl.DataFrame(data, schema=schema, strict=True)

    df = pl.DataFrame(data, schema=schema, strict=False)
    assert df["a"].to_list() == [1, 2, 3]
    assert df["a"].dtype == pl.Int8


def test_df_init_from_series_strict() -> None:
    s = pl.Series("a", [-1, 0, 1])
    schema = {"a": pl.UInt8}
    with pytest.raises(InvalidOperationError):
        pl.DataFrame(s, schema=schema, strict=True)

    df = pl.DataFrame(s, schema=schema, strict=False)
    assert df["a"].to_list() == [None, 0, 1]
    assert df["a"].dtype == pl.UInt8


# https://github.com/pola-rs/polars/issues/15471
def test_df_init_rows_overrides_non_existing() -> None:
    df = pl.DataFrame([{"a": 1}], schema_overrides={"a": pl.Int8(), "b": pl.Boolean()})
    assert df.schema == OrderedDict({"a": pl.Int8})

    df = pl.DataFrame(
        [{"a": 3, "b": 1.0}],
        schema_overrides={"a": pl.Int8, "c": pl.Utf8},
    )
    assert df.schema == OrderedDict({"a": pl.Int8, "b": pl.Float64})


# https://github.com/pola-rs/polars/issues/15245
def test_df_init_nested_mixed_types() -> None:
    data = [{"key": [{"value": 1}, {"value": 1.0}]}]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {"key": pl.List(pl.Struct({"value": pl.Float64}))}
    assert df.to_dicts() == [{"key": [{"value": 1.0}, {"value": 1.0}]}]


@pytest.mark.parametrize(
    ("dtype", "value", "invalid"),
    [
        (pl.Struct({"a": pl.Int64}), {"a": 1}, 5),
        (pl.Struct({"a": pl.Int64}), {"a": 1}, [1, 2]),
        (pl.List(pl.Int64), [1], 5),
        (pl.List(pl.Int64), [1], {"a": 1}),
        (pl.Array(pl.Int64, 1), [1], "x"),
        (pl.Map(pl.String, pl.Int64), {"a": 1}, 5),
    ],
)
@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_nested_strict(
    dtype: pl.DataType, value: Any, invalid: Any, as_dicts: bool
) -> None:
    def init(strict: bool) -> pl.DataFrame:
        if as_dicts:
            data = [{"x": value}, {"x": invalid}]
            return pl.from_dicts(data, schema={"x": dtype}, strict=strict)
        rows = [(value,), (invalid,)]
        return pl.DataFrame(rows, schema={"x": dtype}, orient="row", strict=strict)

    with pytest.raises(ComputeError, match="could not append value"):
        init(strict=True)

    # non-strict, an invalid struct value is null or has all-null fields
    first, second = init(strict=False).to_series().to_list()
    assert first == value
    assert second is None or all(v is None for v in second.values())


@pytest.mark.parametrize(
    ("dtype", "value", "invalid", "non_strict", "error"),
    [
        (pl.Int64, 1, "x", None, ComputeError),
        (pl.Int64, 1, [1], None, ComputeError),
        (pl.Int64, 1, 2**130, None, OverflowError),
        (pl.String, "a", [1], None, ComputeError),
        (pl.Date, date(2020, 1, 1), "x", None, ComputeError),
        (pl.List(pl.Int64), [1], [1, "a"], [1, None], TypeError),
        (pl.Map(pl.String, pl.Int64), {"k": 1}, {"k": "1"}, {"k": 1}, TypeError),
    ],
)
@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_non_strict(
    dtype: pl.DataType,
    value: Any,
    invalid: Any,
    non_strict: Any,
    error: type[Exception],
    as_dicts: bool,
) -> None:
    def init(strict: bool) -> pl.DataFrame:
        if as_dicts:
            data = [{"x": value}, {"x": invalid}]
            return pl.from_dicts(data, schema={"x": dtype}, strict=strict)
        rows = [(value,), (invalid,)]
        return pl.DataFrame(rows, schema={"x": dtype}, orient="row", strict=strict)

    with pytest.raises(error):
        init(strict=True)

    # non-strict, an invalid value is cast or else null
    assert init(strict=False).to_series().to_list() == [value, non_strict]


@pytest.mark.parametrize(
    ("dtype", "lossy", "cast"),
    [
        (pl.Int64, 1.5, 1),
        (pl.Int32, -2.5, -2),
        (pl.UInt8, 2.5, 2),
        (pl.Int64, "1.5", 1),
        (pl.Int64, Decimal("1.5"), 2),
        (pl.Boolean, 2, True),
        (pl.Boolean, 0.5, True),
        (pl.Boolean, "1", None),
        (pl.Boolean, date(1970, 1, 2), None),
        (pl.Date, 1.5, date(1970, 1, 2)),
        (pl.Datetime("us"), 1.5, datetime(1970, 1, 1, 0, 0, 0, 1)),
        (pl.Duration("us"), 1.5, timedelta(microseconds=1)),
        (pl.Time, 1000.5, time(0, 0, 0, 1)),
    ],
)
@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_strict_lossy(
    dtype: pl.DataType, lossy: Any, cast: Any, as_dicts: bool
) -> None:
    def init(strict: bool) -> pl.DataFrame:
        if as_dicts:
            return pl.from_dicts([{"x": lossy}], schema={"x": dtype}, strict=strict)
        rows = [(lossy,)]
        return pl.DataFrame(rows, schema={"x": dtype}, orient="row", strict=strict)

    # a value that would lose information raises...
    with pytest.raises(ComputeError, match="could not append value"):
        init(strict=True)

    # ...and is cast (or else null) if not strict
    assert init(strict=False).to_series().to_list() == [cast]


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        (pl.Int32, 2.0, 2),
        (pl.Int64, True, 1),
        (pl.Int64, Decimal("2.00"), 2),
        (pl.Int64, "-5", -5),
        (pl.UInt64, str(2**64 - 1), 2**64 - 1),
        (pl.Int64, date(1970, 1, 3), 2),
        (pl.Float64, 1, 1.0),
        (pl.Float64, True, 1.0),
        (pl.Float64, Decimal("1.5"), 1.5),
        (pl.Float32, "-5.0", -5.0),
        (pl.Float32, 0.1, 0.10000000149011612),
        (pl.Boolean, 1, True),
        (pl.Boolean, 0.0, False),
        (pl.String, 1, "1"),
        (pl.String, 1.5, "1.5"),
        (pl.String, True, "true"),
        (pl.Date, 2.0, date(1970, 1, 3)),
        (pl.Datetime("ns"), datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 12)),
        (pl.Duration("ms"), timedelta(days=1), timedelta(days=1)),
        (pl.Time, time(1, 2, 3), time(1, 2, 3)),
    ],
)
@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_strict_exact(
    dtype: pl.DataType, value: Any, expected: Any, as_dicts: bool
) -> None:
    # a value that converts exactly is accepted
    if as_dicts:
        df = pl.from_dicts([{"x": value}, {"x": None}], schema={"x": dtype})
    else:
        df = pl.DataFrame([(value,), (None,)], schema={"x": dtype}, orient="row")
    assert df.schema == {"x": dtype}
    assert df.to_series().to_list() == [expected, None]


@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_strict_inferred(as_dicts: bool) -> None:
    def init(values: list[Any], **kwargs: Any) -> pl.Series:
        if as_dicts:
            return pl.DataFrame([{"x": v} for v in values], **kwargs).to_series()
        rows = [(v,) for v in values]
        return pl.DataFrame(rows, orient="row", **kwargs).to_series()

    # values are cast to the inferred supertype
    assert init([1, 2.5]).to_list() == [1.0, 2.5]
    assert init([True, 1]).to_list() == [1, 1]
    assert init([1, "a"]).to_list() == ["1", "a"]

    # past `infer_schema_length`, a lossy value raises (unless not strict)
    assert init([1, 2.0], infer_schema_length=1).to_list() == [1, 2]
    with pytest.raises(ComputeError, match="infer_schema_length"):
        init([1, 2.5], infer_schema_length=1)
    assert init([1, 2.5], infer_schema_length=1, strict=False).to_list() == [1, 2]


@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_non_strict_boolean(as_dicts: bool) -> None:
    def init(values: list[Any]) -> pl.Series:
        schema = {"x": pl.Boolean}
        if as_dicts:
            data = [{"x": v} for v in values]
            return pl.from_dicts(data, schema=schema, strict=False).to_series()
        rows = [(v,) for v in values]
        return pl.DataFrame(rows, schema=schema, orient="row", strict=False).to_series()

    # as with a cast, any nonzero number is true...
    values = [0, 1, 2, 42, -1, 0.5]
    expected = pl.Series("x", values, strict=False).cast(pl.Boolean)
    assert_series_equal(init(values), expected)

    # ...and a string or temporal value cannot be cast, so it is null
    uncastable = ["1", "0", "-5", date(1970, 1, 2), timedelta(1)]
    assert init(uncastable).to_list() == [None] * len(uncastable)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (None, [{"a": 1}, {"a": 2.5}], [{"a": 1.0}, {"a": 2.5}]),
        (pl.Struct({"a": pl.Float64}), [{"a": 1}], [{"a": 1.0}]),
        (pl.Struct({"a": pl.Float16}), [{"a": 1}], [{"a": 1.0}]),
        (
            pl.Struct({"a": pl.Datetime("ms")}),
            [{"a": datetime(2020, 1, 1)}],
            [{"a": datetime(2020, 1, 1)}],
        ),
        (pl.Struct({"a": pl.Int64, "b": pl.Int64}), [(1, 2)], [{"a": 1, "b": 2}]),
        (pl.Decimal(10, 2), [1], [Decimal("1.00")]),
        (None, [[1], [2.5]], [[1.0], [2.5]]),
        (pl.Map(pl.String, pl.Float64), [{"a": 1}, None], [{"a": 1.0}, None]),
    ],
)
@pytest.mark.parametrize("as_dicts", [True, False])
def test_df_init_rows_nested_lenient_values(
    dtype: pl.DataType | None, values: list[Any], expected: list[Any], as_dicts: bool
) -> None:
    # values of the right kind keep their lenient conversion under (default) strict
    schema = {"x": dtype}
    if as_dicts:
        df = pl.from_dicts([{"x": v} for v in values], schema=schema)
    else:
        df = pl.DataFrame([(v,) for v in values], schema=schema, orient="row")
    assert df.to_series().to_list() == expected


class CustomSchema(Mapping[str, Any]):
    """Dummy schema object for testing compatibility with Mapping."""

    _entries: dict[str, Any]

    def __init__(self, **named_entries: Any) -> None:
        self._items = OrderedDict(named_entries.items())

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        yield from self._items


def test_custom_schema() -> None:
    df = pl.DataFrame(schema=CustomSchema(bool=pl.Boolean, misc=pl.UInt8))
    assert df.schema == OrderedDict([("bool", pl.Boolean), ("misc", pl.UInt8)])

    with pytest.raises(TypeError):
        pl.DataFrame(schema=CustomSchema(bool="boolean", misc="unsigned int"))


def test_list_null_constructor_schema() -> None:
    expected = pl.List(pl.Null)
    assert pl.DataFrame({"a": [[]]}).dtypes[0] == expected
    assert pl.DataFrame(schema={"a": pl.List}).dtypes[0] == expected


def test_df_init_schema_object() -> None:
    schema = pl.Schema({"a": pl.Int8(), "b": pl.String()})
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, schema=schema)

    assert df.columns == schema.names()
    assert df.dtypes == schema.dtypes()


def test_df_init_data_orientation_inference_warning() -> None:
    with pytest.warns(DataOrientationWarning):
        pl.from_records([[1, 2, 3], [4, 5, 6]], schema=["a", "b", "c"])


def test_df_init_enum_dtype() -> None:
    class PythonEnum(str, enum.Enum):
        A = "A"
        B = "B"
        C = "C"

    df = pl.DataFrame({"Col 1": ["A", "B", "C"]}, schema={"Col 1": PythonEnum})
    assert df.dtypes[0] == pl.Enum(["A", "B", "C"])


@pytest.mark.parametrize(
    "schema_param",
    [
        {
            "schema": {
                "date": pl.Date,
                "time": pl.Time,
                "datetime": pl.Datetime,
            },
        },
        {
            "schema_overrides": {
                "date": pl.Date(),
                "time": pl.Time(),
                "datetime": pl.Datetime(),
            },
        },
    ],
)
def test_temporal_string_schema_overrides(schema_param: dict[str, SchemaDict]) -> None:
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2025-10-07"],
            "time": ["12:00:00", "13:30:00"],
            "datetime": ["2024-01-01 23:59:59", "2024-01-02T13:30:00.123456"],
        },
        **schema_param,  # type: ignore[arg-type]
    )
    assert df.schema == {
        "date": pl.Date,
        "time": pl.Time,
        "datetime": pl.Datetime("us"),
    }
    assert df.to_dicts() == [
        {
            "date": date(2024, 1, 1),
            "time": time(12, 0),
            "datetime": datetime(2024, 1, 1, 23, 59, 59),
        },
        {
            "date": date(2025, 10, 7),
            "time": time(13, 30),
            "datetime": datetime(2024, 1, 2, 13, 30, 0, 123456),
        },
    ]


def test_bytes_scalar_broadcast_27620() -> None:
    # https://github.com/pola-rs/polars/issues/27620
    result = pl.DataFrame({"a": b"foo", "b": "foo", "c": [10, 20, 30]})
    expected = pl.DataFrame({"a": [b"foo"] * 3, "b": ["foo"] * 3, "c": [10, 20, 30]})
    assert_frame_equal(result, expected)


def test_df_init_from_sequence_of_generators_29121() -> None:
    result = pl.DataFrame([(x for x in (1, 2, 3)), (x for x in (4, 5, 6))])
    expected = pl.DataFrame({"column_0": [1, 2, 3], "column_1": [4, 5, 6]})
    assert_frame_equal(result, expected)

    result = pl.DataFrame(
        [(x for x in (1, 2, 3)), (x for x in (4, 5, 6))],
        schema=["a", "b", "c"],
        orient="row",
    )
    expected = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert_frame_equal(result, expected)


def test_df_init_resolution_leaves_dispatch_registry_alone_29121() -> None:
    SomeRow = dataclasses.make_dataclass("Row29121", [("a", int)])
    registered = set(_sequence_to_pydf_dispatcher.registry)

    pl.DataFrame([SomeRow(a=1)])

    assert set(_sequence_to_pydf_dispatcher.registry) == registered
    assert _resolved_sequence_handlers[SomeRow] is _sequence_of_dataclasses_to_pydf


def test_df_init_dataclass_class_is_not_an_instance() -> None:
    SomeRow = dataclasses.make_dataclass("Row29121Class", [("a", int)])
    _resolved_sequence_handlers.pop(type, None)

    class Unrelated:
        pass

    assert pl.DataFrame([SomeRow]).dtypes == [pl.Object]
    assert pl.DataFrame([Unrelated]).dtypes == [pl.Object]
    assert pl.Series([SomeRow]).dtype == pl.Object

    # instances are still unpacked into columns
    assert pl.DataFrame([SomeRow(a=1)]).to_dict(as_series=False) == {"a": [1]}
    assert pl.Series([SomeRow(a=1)]).dtype == pl.Struct({"a": pl.Int64})


# Type resolution is memoized process-wide, so the first-use window this
# exercises only exists in an interpreter that no other test has warmed up
_CONCURRENT_INIT_SCRIPT = """\
import dataclasses
import sys
import threading

import polars as pl
sys.setswitchinterval(1e-6)

THREADS = 8
ROUNDS = 25
errors = []

def build(payload, barrier):
    barrier.wait()
    try:
        pl.DataFrame(payload)
    except Exception as exc:
        errors.append(exc)

def race(payloads):
    # time out rather than deadlock if a thread never reaches the barrier
    barrier = threading.Barrier(len(payloads), timeout=30)
    threads = [threading.Thread(target=build, args=(p, barrier)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

# the type from the issue report; only races on its very first use in the process
race([[pl.Series(f"c{i}", [i])] for i in range(THREADS)])

# a fresh dataclass type per round reopens the first-use window each time,
# so the race can be exercised repeatedly instead of getting a single shot
for r in range(ROUNDS):
    Row = dataclasses.make_dataclass(f"Row{r}", [("a", int)])
    race([[Row(a=i)] for i in range(THREADS)])

assert not errors, f"{len(errors)} constructions failed, first: {errors[0]!r}"
"""


def test_df_init_concurrent_first_use_29121() -> None:
    # resolving a type for the first time must not mutate
    # state that a concurrent dispatch is reading
    proc = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_INIT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
