from __future__ import annotations

from collections import UserDict
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


def test_from_dict_with_column_order() -> None:
    # expect schema/columns order to take precedence
    schema = {"a": pl.UInt8, "b": pl.UInt32}
    data = {"b": [3, 4], "a": [1, 2]}
    for df in (
        pl.DataFrame(data, schema=schema),
        pl.DataFrame(data, schema=["a", "b"], schema_overrides=schema),
    ):
        # ┌─────┬─────┐
        # │ a   ┆ b   │
        # │ --- ┆ --- │
        # │ u8  ┆ u32 │
        # ╞═════╪═════╡
        # │ 1   ┆ 3   │
        # │ 2   ┆ 4   │
        # └─────┴─────┘
        assert df.columns == ["a", "b"]
        assert df.schema == {"a": pl.UInt8, "b": pl.UInt32}
        assert df.rows() == [(1, 3), (2, 4)]

        # expect an error
        mismatched_schema = {"x": pl.UInt8, "b": pl.UInt32}
        with pytest.raises(ValueError):
            pl.DataFrame({"b": [3, 4], "a": [1, 2]}, schema=mismatched_schema)


def test_from_dict_with_scalars() -> None:
    # one or more valid arrays, with some scalars (inc. None)
    df1 = pl.DataFrame(
        {"key": ["aa", "bb", "cc"], "misc": "xyz", "other": None, "value": 0}
    )
    assert df1.to_dict(as_series=False) == {
        "key": ["aa", "bb", "cc"],
        "misc": ["xyz", "xyz", "xyz"],
        "other": [None, None, None],
        "value": [0, 0, 0],
    }

    # edge-case: all scalars
    df2 = pl.DataFrame({"key": "aa", "misc": "xyz", "other": None, "value": 0})
    assert df2.to_dict(as_series=False) == {
        "key": ["aa"],
        "misc": ["xyz"],
        "other": [None],
        "value": [0],
    }

    # edge-case: single unsized generator
    df3 = pl.DataFrame({"vals": map(float, [1, 2, 3])})
    assert df3.to_dict(as_series=False) == {"vals": [1.0, 2.0, 3.0]}

    # ensure we don't accidentally consume or expand map/range/generator
    # cols, and can properly apply schema dtype/ordering directives
    df4 = pl.DataFrame(
        {
            "key": range(1, 4),
            "misc": (x for x in [4, 5, 6]),
            "other": map(float, [7, 8, 9]),
            "value": {0: "x", 1: "y", 2: "z"}.values(),
        },
        schema={
            "value": pl.String,
            "other": pl.Float32,
            "misc": pl.Int32,
            "key": pl.Int8,
        },
    )
    assert df4.columns == ["value", "other", "misc", "key"]
    assert df4.to_dict(as_series=False) == {
        "value": ["x", "y", "z"],
        "other": [7.0, 8.0, 9.0],
        "misc": [4, 5, 6],
        "key": [1, 2, 3],
    }
    assert df4.schema == {
        "value": pl.String,
        "other": pl.Float32,
        "misc": pl.Int32,
        "key": pl.Int8,
    }

    # mixed with struct cols
    for df5 in (
        pl.from_dict(
            {"x": {"b": [1, 3], "c": [2, 4]}, "y": [5, 6], "z": "x"},
            schema_overrides={"y": pl.Int8},
        ),
        pl.from_dict(
            {"x": {"b": [1, 3], "c": [2, 4]}, "y": [5, 6], "z": "x"},
            schema=["x", ("y", pl.Int8), "z"],
        ),
    ):
        assert df5.rows() == [({"b": 1, "c": 2}, 5, "x"), ({"b": 3, "c": 4}, 6, "x")]
        assert df5.schema == {
            "x": pl.Struct([pl.Field("b", pl.Int64), pl.Field("c", pl.Int64)]),
            "y": pl.Int8,
            "z": pl.String,
        }

    # mixed with numpy cols...
    df6 = pl.DataFrame(
        {"x": np.ones(3), "y": np.zeros(3), "z": 1.0},
    )
    assert df6.rows() == [(1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

    # ...and trigger multithreaded load codepath
    df7 = pl.DataFrame(
        {
            "w": np.zeros(1001, dtype=np.uint8),
            "x": np.ones(1001, dtype=np.uint8),
            "y": np.zeros(1001, dtype=np.uint8),
            "z": 1,
        },
        schema_overrides={"z": pl.UInt8},
    )
    assert df7[999:].rows() == [(0, 1, 0, 1), (0, 1, 0, 1)]
    assert df7.schema == {
        "w": pl.UInt8,
        "x": pl.UInt8,
        "y": pl.UInt8,
        "z": pl.UInt8,
    }

    # misc generators/iterables
    df9 = pl.DataFrame(
        {
            "a": iter([0, 1, 2]),
            "b": (2, 1, 0).__iter__(),
            "c": (v for v in (0, 0, 0)),
            "d": "x",
        }
    )
    assert df9.rows() == [(0, 2, 0, "x"), (1, 1, 0, "x"), (2, 0, 0, "x")]


@pytest.mark.slow
def test_from_dict_with_values_mixed() -> None:
    # a bit of everything
    mixed_dtype_data: dict[str, Any] = {
        "a": 0,
        "b": 8,
        "c": 9.5,
        "d": None,
        "e": True,
        "f": False,
        "g": time(0, 1, 2),
        "h": date(2023, 3, 14),
        "i": timedelta(seconds=3601),
        "j": datetime(2111, 11, 11, 11, 11, 11, 11),
        "k": "「趣味でヒーローをやっている者だ」",
    }
    # note: deliberately set this value large; if all dtypes are
    # on the fast-path it'll only take ~0.03secs. if it becomes
    # even remotely noticeable that will indicate a regression.
    n_range = 1_000_000
    index_and_data: dict[str, Any] = {"idx": range(n_range)}
    index_and_data.update(mixed_dtype_data.items())
    df = pl.DataFrame(
        data=index_and_data,
        schema={
            "idx": pl.Int32,
            "a": pl.UInt16,
            "b": pl.UInt32,
            "c": pl.Float64,
            "d": pl.Float32,
            "e": pl.Boolean,
            "f": pl.Boolean,
            "g": pl.Time,
            "h": pl.Date,
            "i": pl.Duration,
            "j": pl.Datetime,
            "k": pl.String,
        },
    )
    dfx = df.select(pl.exclude("idx"))

    assert df.height == n_range
    assert dfx[:5].rows() == dfx[5:10].rows()
    assert dfx[-10:-5].rows() == dfx[-5:].rows()
    assert dfx.row(n_range // 2, named=True) == mixed_dtype_data


def test_from_dict_expand_nested_struct() -> None:
    # confirm consistent init of nested struct from dict data
    dt = date(2077, 10, 10)
    expected = pl.DataFrame(
        [
            pl.Series("x", [dt]),
            pl.Series("nested", [{"y": -1, "z": 1}]),
        ]
    )
    for df in (
        pl.DataFrame({"x": dt, "nested": {"y": -1, "z": 1}}),
        pl.DataFrame({"x": dt, "nested": [{"y": -1, "z": 1}]}),
        pl.DataFrame({"x": [dt], "nested": {"y": -1, "z": 1}}),
        pl.DataFrame({"x": [dt], "nested": [{"y": -1, "z": 1}]}),
    ):
        assert_frame_equal(expected, df)

    # confirm expansion to 'n' nested values
    nested_values = [{"y": -1, "z": 1}, {"y": -1, "z": 1}, {"y": -1, "z": 1}]
    expected = pl.DataFrame(
        [
            pl.Series("x", [0, 1, 2]),
            pl.Series("nested", nested_values),
        ]
    )
    for df in (
        pl.DataFrame({"x": range(3), "nested": {"y": -1, "z": 1}}),
        pl.DataFrame({"x": [0, 1, 2], "nested": {"y": -1, "z": 1}}),
    ):
        assert_frame_equal(expected, df)


def test_from_dict_duration_subseconds() -> None:
    d = {"duration": [timedelta(seconds=1, microseconds=1000)]}
    result = pl.from_dict(d)
    expected = pl.select(duration=pl.duration(seconds=1, microseconds=1000))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("dtype", "data"),
    [
        (pl.Date, date(2099, 12, 31)),
        (pl.Datetime("ms"), datetime(1998, 10, 1, 10, 30)),
        (pl.Duration("us"), timedelta(days=1)),
        (pl.Time, time(2, 30, 10)),
    ],
)
def test_from_dict_cast_logical_type(dtype: pl.DataType, data: Any) -> None:
    schema = {"data": dtype}
    df = pl.DataFrame({"data": [data]}, schema=schema)
    physical_dict = df.cast(pl.Int64).to_dict()

    df_from_dicts = pl.from_dicts(
        [
            {
                "data": physical_dict["data"][0],
            }
        ],
        schema=schema,
    )

    assert_frame_equal(df_from_dicts, df)


def test_from_dicts_mixed_mappings() -> None:
    result = pl.from_dicts([{"a": 1}, UserDict({"a": 2, "b": 3}), {"b": 4}])
    expected = pl.DataFrame({"a": [1, 2, None], "b": [None, 3, 4]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "schema", "infer_schema_length", "value"),
    [
        ([{"x": 1}], None, 0, "1 of type: i64"),
        ([{"x": 1}] * 3 + [{"x": "a"}], None, 3, '"a" of type: str'),
        ([{"x": 1}, {"x": 1000}], {"x": pl.Int8}, 100, "1000 of type: i64"),
    ],
)
def test_from_dicts_rejected_value(
    data: list[dict[str, Any]],
    schema: dict[str, pl.DataType] | None,
    infer_schema_length: int,
    value: str,
) -> None:
    with pytest.raises(ComputeError, match=f"could not append value: {value}"):
        pl.from_dicts(data, schema=schema, infer_schema_length=infer_schema_length)


@pytest.mark.parametrize("schema_mode", ["infer", "schema", "overrides"])
@pytest.mark.parametrize("strict", [True, False])
def test_from_dicts_string_float_values(schema_mode: str, strict: bool) -> None:
    class StringSubclass(str):
        pass

    class FloatSubclass(float):
        pass

    strings = ["", "a" * 12, "b" * 13, "c" * 24, "é漢🙂" * 100, "a\0b"]
    numbers = [-0.0, 0.0, float("inf"), float("-inf"), float("nan"), 1.25]
    data: list[Any] = [{"text": "prefix", "number": 0.5}]
    data.extend({"number": n, "text": s} for s, n in zip(strings, numbers, strict=True))
    data.extend(
        [
            {"text": StringSubclass("subclass"), "number": FloatSubclass(2.5)},
            {"text": None, "number": None},
            {},
            None,
            {"number": 3.5, "text": "last", "ignored": "extra"},
        ]
    )
    schema = {"text": pl.String, "number": pl.Float64}
    result = pl.from_dicts(
        data,
        schema=schema if schema_mode == "schema" else None,
        schema_overrides=schema if schema_mode == "overrides" else None,
        infer_schema_length=1,
        strict=strict,
    )
    expected = pl.DataFrame(
        {
            "text": ["prefix", *strings, "subclass", None, None, None, "last"],
            "number": [0.5, *numbers, 2.5, None, None, None, 3.5],
        },
        schema=schema,
    )
    assert_frame_equal(result, expected)
    assert np.signbit(result["number"][1])


@pytest.mark.parametrize("strict", [True, False])
def test_from_dicts_string_float_dtype_fallback(strict: bool) -> None:
    result = pl.from_dicts(
        [{"text": "a", "number": 1.5}, {"text": "b", "number": 2}],
        schema={"text": pl.Categorical, "number": pl.Float32},
        strict=strict,
    )
    expected = pl.DataFrame(
        {"text": ["a", "b"], "number": [1.5, 2.0]},
        schema={"text": pl.Categorical, "number": pl.Float32},
    )
    assert_frame_equal(result, expected)


def test_from_dicts_mapping_error_lookup_order() -> None:
    calls: list[tuple[str, str]] = []

    class TrackedMapping(UserDict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            calls.append((self.data["text"], key))
            return super().__getitem__(key)

    data: list[Any] = [
        {"text": "prefix", "number": 1.0},
        TrackedMapping({"text": "valid", "number": 2.0}),
        TrackedMapping({"text": "invalid", "number": []}),
        TrackedMapping({"text": "unread", "number": 3.0}),
    ]
    with pytest.raises(ComputeError, match="could not append value"):
        pl.from_dicts(data, schema={"text": pl.String, "number": pl.Float64})
    assert calls == [
        ("valid", "text"),
        ("valid", "number"),
        ("invalid", "text"),
        ("invalid", "number"),
        ("invalid", "number"),
    ]


@pytest.mark.parametrize("infer_schema_length", [1, None])
def test_from_dicts_invalid_unicode(infer_schema_length: int | None) -> None:
    with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
        pl.from_dicts(
            [{"text": "valid"}, {"text": "\ud800"}],
            infer_schema_length=infer_schema_length,
        )


@pytest.mark.parametrize("strict", [True, False])
def test_from_dicts_string_float_value_fallback(strict: bool) -> None:
    result = pl.from_dicts(
        [{"text": "first", "number": 1.5}, {"text": 42, "number": 2}],
        infer_schema_length=1,
        strict=strict,
    )
    expected = pl.DataFrame({"text": ["first", "42"], "number": [1.5, 2.0]})
    assert_frame_equal(result, expected)
