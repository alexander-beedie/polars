from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_reverse_list_22829() -> None:
    schema = {"a": pl.Int64(), "b": pl.List(pl.Binary)}
    df = pl.DataFrame([], schema=schema, schema_overrides=schema)
    assert df.schema == df.reverse().schema


def test_reverse_df() -> None:
    out = pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).reverse()
    expected = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    assert_frame_equal(out.collect(), expected)


def test_reverse_series() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5])
    assert s.reverse().to_list() == [5, 4, 3, 2, 1]

    s = pl.Series("values", ["a", "b", None, "y", "x"])
    assert s.reverse().to_list() == ["x", "y", None, "b", "a"]


def test_reverse_binary() -> None:
    # single chunk
    s = pl.Series("values", ["a", "b", "c", "d"]).cast(pl.Binary)
    assert s.reverse().to_list() == [b"d", b"c", b"b", b"a"]

    # multiple chunks
    chunk1 = pl.Series("values", ["a", "b"])
    chunk2 = pl.Series("values", ["c", "d"])
    s = chunk1.extend(chunk2).cast(pl.Binary)
    assert s.n_chunks() == 2
    assert s.reverse().to_list() == [b"d", b"c", b"b", b"a"]


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([[1, None], None, [3, 4], [5, 6]], pl.Array(pl.Int64, 2)),
        ([["a", None], None, ["c", "d"], ["e", "f"]], pl.Array(pl.String, 2)),
        ([[True, None], None, [False, True], [True, True]], pl.Array(pl.Boolean, 2)),
        ([[[1]], None, [[None]], [[4]]], pl.Array(pl.Int64, (1, 1))),
        (
            [[{"a": 1}, None], None, [{"a": None}, {"a": 4}], [{"a": 5}, {"a": 6}]],
            pl.Array(pl.Struct({"a": pl.Int64}), 2),
        ),
        ([[[1], None], None, [[], [2, 3]], [[4], [5]]], pl.Array(pl.List(pl.Int64), 2)),
        (
            [
                [date(2020, 1, 1), None],
                None,
                [date(2020, 1, 3), date(2020, 1, 4)],
                [date(2020, 1, 5), date(2020, 1, 6)],
            ],
            pl.Array(pl.Date, 2),
        ),
        ([[], None, [], []], pl.Array(pl.String, 0)),
    ],
)
def test_reverse_array(values: list[Any], dtype: pl.DataType) -> None:
    s = pl.Series("x", values, dtype=dtype)
    multi_chunk = pl.concat([s, s.slice(1, 2)], rechunk=False)
    assert multi_chunk.n_chunks() == 2

    for s_in, expected in (
        (s, values[::-1]),
        (s.slice(1, 3), values[1:4][::-1]),
        (multi_chunk, (values + values[1:3])[::-1]),
    ):
        assert_series_equal(s_in.reverse(), pl.Series("x", expected, dtype=dtype))
