from __future__ import annotations

import io
from datetime import date
from typing import Any

import pyarrow as pa
import pytest

import polars as pl
from polars.testing import assert_series_equal


@pytest.mark.parametrize("namespace", ["list", "arr"])
@pytest.mark.parametrize("depth", [0, 1])
def test_unnest_namespace_guard(namespace: str, depth: int) -> None:
    for value in (5, None):
        expr = getattr(pl.lit(value), namespace)
        with pytest.raises(pl.exceptions.InvalidOperationError):
            pl.select(expr.unnest(depth))


def test_list_unnest_depth_and_nulls() -> None:
    s = pl.Series(
        "x",
        [[[1, None], None, [2]], [], None, [[3]]],
        dtype=pl.List(pl.List(pl.Int64)),
    )
    expected = pl.Series(
        "x", [[1, None, None, 2], [], None, [3]], dtype=pl.List(pl.Int64)
    )
    assert_series_equal(s.list.unnest(), expected)
    assert_series_equal(pl.select(pl.lit(s).list.unnest()).to_series(), expected)
    assert_series_equal(s.list.unnest(0), s)

    with pytest.raises(ValueError, match="non-negative"):
        pl.col("x").list.unnest(-1)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="depth"):
        pl.select(pl.lit(s).list.unnest(2))


def test_list_unnest_strings_and_depth_two_null_expansion() -> None:
    df = pl.DataFrame(
        {"data": [[["abc", "ghi"], ["jkl", "lmn"]], [["bcd", "zxc"]], [["def"]]]}
    )
    assert df.select(pl.col("data").list.unnest()).to_series().to_list() == [
        ["abc", "ghi", "jkl", "lmn"],
        ["bcd", "zxc"],
        ["def"],
    ]

    nested = pl.Series(
        "x",
        [[None], [[[1, 2]]]],
        dtype=pl.List(pl.List(pl.Array(pl.Int64, 2))),
    )
    assert_series_equal(
        nested.list.unnest(2),
        pl.Series("x", [[None, None], [1, 2]], dtype=pl.List(pl.Int64)),
    )


def test_array_unnest_partial_and_full_depth() -> None:
    dtype = pl.Array(pl.Array(pl.Array(pl.Int64, 2), 3), 2)
    s = pl.Series(
        "x",
        [[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]],
        dtype=dtype,
    )
    assert_series_equal(s.arr.unnest(0), s)
    assert_series_equal(
        s.arr.unnest(),
        pl.Series(
            "x",
            [[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]],
            dtype=pl.Array(pl.Array(pl.Int64, 2), 6),
        ),
    )
    assert_series_equal(
        s.arr.unnest(2),
        pl.Series(
            "x", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=pl.Array(pl.Int64, 12)
        ),
    )


def test_array_unnest_inner_nulls_and_mixed_list() -> None:
    arrays = pl.Series(
        "x",
        [[[1, None], None], None, [[3, 4], [5, 6]]],
        dtype=pl.Array(pl.Array(pl.Int64, 2), 2),
    )
    assert_series_equal(
        arrays.arr.unnest(),
        pl.Series(
            "x",
            [[1, None, None, None], None, [3, 4, 5, 6]],
            dtype=pl.Array(pl.Int64, 4),
        ),
    )

    mixed = pl.Series(
        "x", [[[1, 2], None], None, [[], [3]]], dtype=pl.Array(pl.List(pl.Int64), 2)
    )
    assert_series_equal(
        mixed.arr.unnest(),
        pl.Series("x", [[1, 2, None], None, [3]], dtype=pl.List(pl.Int64)),
    )


def test_array_unnest_zero_width_preserves_rows() -> None:
    s = pl.Series(
        "x", [[[], []], None, [[], []]], dtype=pl.Array(pl.Array(pl.Int64, 0), 2)
    )
    assert_series_equal(
        s.arr.unnest(),
        pl.Series("x", [[], None, []], dtype=pl.Array(pl.Int64, 0)),
    )


def test_unnest_sliced_multichunk_and_logical_leaf() -> None:
    d1 = date(2024, 1, 1)
    d2 = date(2024, 1, 2)
    first = pl.Series("x", [[[d1], [d2]], None], dtype=pl.List(pl.List(pl.Date)))
    second = pl.Series("x", [[[d2]], []], dtype=pl.List(pl.List(pl.Date)))
    s = pl.concat([first, second], rechunk=False).slice(1, 3)
    assert s.n_chunks() > 1
    assert_series_equal(
        s.list.unnest(),
        pl.Series("x", [None, [d2], []], dtype=pl.List(pl.Date)),
    )


def test_unnest_lazy_streaming_grouped_and_serde() -> None:
    df = pl.DataFrame({"g": [1, 1, 2], "x": [[[1], [2]], [[3]], [[4]]]})
    expr = pl.col("x").list.unnest()
    lazy = df.lazy().select(expr)
    assert lazy.collect_schema()["x"] == pl.List(pl.Int64)
    expected = df.select(expr)
    assert lazy.collect(engine="streaming").equals(expected)
    assert df.group_by("g", maintain_order=True).agg(expr).to_dict(as_series=False) == {
        "g": [1, 2],
        "x": [[[1, 2], [3]], [[4]]],
    }
    restored = pl.Expr.deserialize(io.BytesIO(expr.meta.serialize()))
    assert df.select(restored).equals(expected)


def test_list_unnest_skips_values_hidden_under_nulls() -> None:
    # Null lists may still span values in Arrow; they must not leak into the output.
    inner = pa.LargeListArray.from_arrays(
        pa.array([0, 2, 5, 5, 9, 12], pa.int64()),
        pa.array(range(12), pa.int64()),
        mask=pa.array([False, True, False, True, False]),
    )
    outer = pa.LargeListArray.from_arrays(
        pa.array([0, 2, 4, 5], pa.int64()),
        inner,
        mask=pa.array([False, True, False]),
    )
    s = pl.Series("x", outer)
    expected = pl.Series(
        "x", [[0, 1, None], None, [9, 10, 11]], dtype=pl.List(pl.Int64)
    )
    assert_series_equal(s.list.unnest(), expected)
    assert_series_equal(s.slice(1).list.unnest(), expected.slice(1))


def test_array_unnest_nulls_at_every_level() -> None:
    dtype = pl.Array(pl.Array(pl.Array(pl.Int64, 2), 2), 2)
    s = pl.Series(
        "x",
        [
            [[[1, None], None], None],
            None,
            [[[5, 6], [7, 8]], [None, [None, 9]]],
        ],
        dtype=dtype,
    )
    expected = pl.Series(
        "x",
        [
            [1, None, None, None, None, None, None, None],
            None,
            [5, 6, 7, 8, None, None, None, 9],
        ],
        dtype=pl.Array(pl.Int64, 8),
    )
    assert_series_equal(s.arr.unnest(2), expected)
    assert_series_equal(s.arr.unnest().arr.unnest(), expected)
    assert_series_equal(s.slice(2).arr.unnest(2), expected.slice(2))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Nulls only in the Array level: masked in place.
        ([[[1, 2], None], [[3, 4]]], [1, 2, None, None, 3, 4]),
        # A null List above the Array level: copied, with one null Array of two nulls.
        ([[[1, 2]], None, [[3, 4], None]], [1, 2, None, None, 3, 4, None, None]),
    ],
)
def test_list_unnest_array_below_list(value: list[Any], expected: list[Any]) -> None:
    s = pl.Series("x", [value, None], dtype=pl.List(pl.List(pl.Array(pl.Int64, 2))))
    out = s.list.unnest(2)
    assert_series_equal(out, pl.Series("x", [expected, None], dtype=pl.List(pl.Int64)))
    assert_series_equal(out, s.list.unnest().list.unnest())


def test_unnest_scalar_input() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = df.select(pl.lit([[1], None, [2, 3]]).list.unnest())
    assert out.to_series().to_list() == [[1, None, 2, 3]]
    out = df.with_columns(x=pl.lit([[1], None, [2, 3]]).list.unnest())
    assert out["x"].to_list() == [[1, None, 2, 3]] * 3


def test_array_unnest_null_inner_lists_export_to_arrow() -> None:
    s = pl.Series(
        "x",
        [[None], None, [[]], [None], None],
        dtype=pl.Array(pl.List(pl.Array(pl.Categorical(), 3)), 1),
    )
    out = s.arr.unnest()
    assert out.to_list() == [[None], None, [], [None], None]
    assert_series_equal(pl.Series("x", out.to_arrow()), out)
