from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (a,c)",
            pl.DataFrame({"a": [2], "b": [0], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a SEMI JOIN tbl_b USING (a,c)",
            pl.DataFrame({"a": [2], "b": [0], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (a)",
            pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT ANTI JOIN tbl_b USING (a)",
            pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "c": pl.String}),
        ),
        (
            "SELECT * FROM tbl_a ANTI JOIN tbl_b USING (a)",
            pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "c": pl.String}),
        ),
        (
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (b) LEFT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"a": [1, 3], "b": [4, 6], "c": ["w", "z"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT ANTI JOIN tbl_b USING (b) LEFT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"a": [2], "b": [0], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a RIGHT ANTI JOIN tbl_b USING (b) LEFT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"a": [2], "b": [5], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a RIGHT SEMI JOIN tbl_b USING (b) RIGHT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"c": ["z"], "d": [25.5]}),
        ),
        (
            "SELECT * FROM tbl_a RIGHT SEMI JOIN tbl_b USING (b) RIGHT ANTI JOIN tbl_c USING (c)",
            pl.DataFrame({"c": ["w", "y"], "d": [10.5, -50.0]}),
        ),
    ],
)
def test_join_anti_semi(sql: str, expected: pl.DataFrame) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    ctx = pl.SQLContext(frames, eager=True)
    assert_frame_equal(expected, ctx.execute(sql))


def test_join_right_semi_anti_on_clause() -> None:
    df1 = pl.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "y": [200, 300, 400]})

    ctx = pl.SQLContext({"df1": df1, "df2": df2}, eager=True)

    # RIGHT SEMI → keeps right rows that match
    result = ctx.execute("""
        SELECT df2.id, df2.y
        FROM df1 RIGHT SEMI JOIN df2 ON df1.id = df2.id
        ORDER BY df2.id
    """)
    assert_frame_equal(
        result,
        pl.DataFrame({"id": [2, 3], "y": [200, 300]}),
        check_dtypes=False,
    )

    # RIGHT ANTI → keeps right rows that DON'T match
    result = ctx.execute("""
        SELECT df2.id, df2.y
        FROM df1 RIGHT ANTI JOIN df2 ON df1.id = df2.id
        ORDER BY df2.id
    """)
    assert_frame_equal(
        result,
        pl.DataFrame({"id": [4], "y": [400]}),
        check_dtypes=False,
    )


def test_join_on_parenthesized_condition() -> None:
    assert_sql_matches(
        frames={
            "df1": pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}),
            "df2": pl.DataFrame({"a": [2, 3, 4], "c": ["x", "y", "z"]}),
        },
        query="""
            SELECT df1.a, df1.b, df2.c
            FROM df1 INNER JOIN df2 ON (df1.a = df2.a)
            ORDER BY df1.a
        """,
        compare_with="sqlite",
        expected={"a": [2, 3], "b": [20, 30], "c": ["x", "y"]},
    )


def test_join_cross() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
    }
    with pl.SQLContext(frames, eager=True) as ctx:
        out = ctx.execute(
            """
            SELECT *
            FROM tbl_a
            CROSS JOIN tbl_b
            ORDER BY a, b, c
            """
        )
        assert out.rows() == [
            (1, 4, "w", 3, 6, "x"),
            (1, 4, "w", 2, 5, "y"),
            (1, 4, "w", 1, 4, "z"),
            (2, 0, "y", 3, 6, "x"),
            (2, 0, "y", 2, 5, "y"),
            (2, 0, "y", 1, 4, "z"),
            (3, 6, "z", 3, 6, "x"),
            (3, 6, "z", 2, 5, "y"),
            (3, 6, "z", 1, 4, "z"),
        ]


def test_join_cross_11927() -> None:
    df1 = pl.DataFrame({"id": [1, 2, 3]})
    df2 = pl.DataFrame({"id": [3, 4, 5]})
    df3 = pl.DataFrame({"id": [4, 5, 6]})

    res = pl.sql("SELECT df1.id FROM df1 CROSS JOIN df2 WHERE df1.id = df2.id")
    assert_frame_equal(res.collect(), pl.DataFrame({"id": [3]}))

    res = pl.sql("SELECT * FROM df1 CROSS JOIN df3 WHERE df1.id = df3.id")
    assert res.collect().is_empty()


def test_cross_join_unnest_from_table() -> None:
    df = pl.DataFrame({"id": [1, 2], "items": [[100, 200], [300, 400, 500]]})
    assert_sql_matches(
        frames=df,
        query="""
            SELECT id, item
            FROM self CROSS JOIN UNNEST(items) AS item
            ORDER BY id DESC, item ASC
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 2, 2, 1, 1],
            "item": [300, 400, 500, 100, 200],
        },
    )


def test_cross_join_unnest_from_cte() -> None:
    assert_sql_matches(
        {},
        query="""
            WITH data AS (
                SELECT 'xyz' AS id, [0,1,2] AS items
                UNION ALL
                SELECT 'abc', [3,4]
            )
            SELECT id, item
            FROM data CROSS JOIN UNNEST(items) AS item
            ORDER BY item
        """,
        compare_with="duckdb",
        expected={
            "id": ["xyz", "xyz", "xyz", "abc", "abc"],
            "item": [0, 1, 2, 3, 4],
        },
    )


@pytest.mark.parametrize(
    "join_clause",
    [
        "ON f1.category = f2.category",
        "ON f2.category = f1.category",
        "USING (category)",
    ],
)
def test_join_inner(foods_ipc_path: Path, join_clause: str) -> None:
    foods1 = pl.scan_ipc(foods_ipc_path)
    foods2 = foods1
    schema = foods1.collect_schema()

    out = pl.sql(
        f"""
        SELECT *
        FROM
          (SELECT * FROM foods1 WHERE fats_g != 0) f1
        INNER JOIN
          (SELECT * FROM foods2 WHERE fats_g = 0) f2
        {join_clause}
        ORDER BY ALL
        LIMIT 2
        """,
        eager=True,
    )
    expected = pl.DataFrame(
        {
            "category": ["fruit", "fruit"],
            "calories": [50, 50],
            "fats_g": [4.5, 4.5],
            "sugars_g": [0, 0],
            "category:f2": ["fruit", "fruit"],
            "calories:f2": [30, 30],
            "fats_g:f2": [0.0, 0.0],
            "sugars_g:f2": [3, 5],
        }
    )
    assert_frame_equal(expected, out, check_dtypes=False)


def test_join_inner_15663() -> None:
    df_a = pl.DataFrame({"LOCID": [1, 2, 3], "VALUE": [0.1, 0.2, 0.3]})
    df_b = pl.DataFrame({"LOCID": [1, 2, 3], "VALUE": [25.6, 53.4, 12.7]})
    df_expected = pl.DataFrame(
        {
            "LOCID": [1, 2, 3],
            "VALUE_A": [0.1, 0.2, 0.3],
            "VALUE_B": [25.6, 53.4, 12.7],
        }
    )
    with pl.SQLContext(register_globals=True, eager=True) as ctx:
        query = """
            SELECT
                a.LOCID,
                a.VALUE AS VALUE_A,
                b.VALUE AS VALUE_B
            FROM df_a AS a INNER JOIN df_b AS b USING (LOCID)
            ORDER BY LOCID
        """
        actual = ctx.execute(query)
        assert_frame_equal(df_expected, actual)


@pytest.mark.parametrize(
    ("join_clause", "expected_error"),
    [
        (
            """
            INNER JOIN tbl_b USING (a,b)
            INNER JOIN tbl_c USING (c)
            """,
            None,
        ),
        (
            """
            INNER JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
            INNER JOIN tbl_c ON tbl_b.c = tbl_c.c
            """,
            None,
        ),
        (
            """
            INNER JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
            INNER JOIN tbl_c ON tbl_a.c = tbl_c.c  --<< (no "c" in 'tbl_a')
            """,
            "no column named 'c' found in table 'tbl_a'",
        ),
    ],
)
def test_join_inner_multi(join_clause: str, expected_error: str | None) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        assert ctx.tables() == ["tbl_a", "tbl_b", "tbl_c"]
        query = f"""
            SELECT tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d
            FROM tbl_a {join_clause}
            ORDER BY tbl_a.a DESC
        """
        try:
            out = ctx.execute(query)
            assert out.collect().rows() == [(1, 4, "z", 25.5)]

        except SQLInterfaceError as err:
            if not (expected_error and expected_error in str(err)):
                raise


@pytest.mark.parametrize(
    "join_clause",
    [
        """
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        """,
        """
        LEFT JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
        LEFT JOIN tbl_c ON tbl_b.c = tbl_c.c
        """,
    ],
)
def test_join_left_multi(join_clause: str) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        for select_cols in (
            "tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d",
            "tbl_a.a, tbl_a.b, tbl_b.c, d",
        ):
            out = ctx.execute(
                f"SELECT {select_cols} FROM tbl_a {join_clause} ORDER BY a DESC"
            )
            assert out.collect().rows() == [
                (3, 6, "x", None),
                (2, None, None, None),
                (1, 4, "z", 25.5),
            ]


def test_join_left_multi_nested() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        out = ctx.execute(
            """
            SELECT tbl_x.a, tbl_x.b, tbl_x.c, tbl_c.d FROM (
                SELECT *
                FROM tbl_a
                LEFT JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
            ) tbl_x
            LEFT JOIN tbl_c ON tbl_x.c = tbl_c.c
            ORDER BY tbl_x.a ASC
            """
        ).collect()

        assert out.rows() == [
            (1, 4, "z", 25.5),
            (2, None, None, None),
            (3, 6, "x", None),
        ]


def test_join_misc_13618() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )
    res = (
        pl.SQLContext(t=df, t1=df, eager=True)
        .execute(
            """
            SELECT t.A, t.fruits, t1.B, t1.cars
            FROM t
            JOIN t1 ON t.A = t1.B
            ORDER BY t.A DESC
            """
        )
        .to_dict(as_series=False)
    )
    assert res == {
        "A": [5, 4, 3, 2, 1],
        "fruits": ["banana", "apple", "apple", "banana", "banana"],
        "B": [5, 4, 3, 2, 1],
        "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
    }


def test_join_misc_16255() -> None:
    df1 = pl.read_csv(BytesIO(b"id,data\n1,open"))
    df2 = pl.read_csv(BytesIO(b"id,data\n1,closed"))
    res = pl.sql(
        """
        SELECT a.id, a.data AS d1, b.data AS d2
        FROM df1 AS a JOIN df2 AS b
        ON a.id = b.id
        """,
        eager=True,
    )
    assert res.rows() == [(1, "open", "closed")]


def test_implicit_join() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 3, 4, 4, 5]})
    df2 = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0, 3, 4, 5, 6]})
    df3 = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [0, 3, 4, 5, 7], "c": [1, 3, 4, 5, 7]}
    )
    frames: dict[str, pl.DataFrame | pl.LazyFrame] = {
        "df1": df1,
        "df2": df2,
        "df3": df3,
    }

    # two tables
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT df1.a, df1.b, df2.b AS b2
            FROM df1, df2
            WHERE df1.a = df2.a AND df1.b = df2.b
        """,
        compare_with="sqlite",
        expected={"a": [2, 3], "b": [3, 4], "b2": [3, 4]},
    )

    # three tables (varying FROM order)
    for tables in ("df1, df2, df3", "df3, df2, df1", "df2, df3, df1"):
        assert_sql_matches(
            frames=frames,
            query=f"""
                SELECT df1.a, df1.b, df3.c
                FROM {tables}
                WHERE df1.a = df2.a
                  AND df3.a = df1.a
                  AND df1.b = df2.b
                  AND df3.b = df1.b
            """,
            compare_with="sqlite",
            expected={"a": [2, 3], "b": [3, 4], "c": [3, 4]},
        )

    # three tables, third references second table
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT df1.a, df1.b, df3.c
            FROM df1, df2, df3
            WHERE df1.a = df2.a AND df1.b = df2.b
              AND df3.a = df2.a AND df3.b = df2.b
        """,
        compare_with="sqlite",
        expected={"a": [2, 3], "b": [3, 4], "c": [3, 4]},
    )

    # with residual (non-join) filter
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT df1.a, df1.b, df2.b AS b2
            FROM df1, df2
            WHERE df1.a = df2.a AND df1.a > 2
        """,
        compare_with="sqlite",
        expected={"a": [3, 4, 5], "b": [4, 4, 5], "b2": [4, 5, 6]},
    )

    # mixed implicit + explicit join
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT df1.a, df1.b, df3.c
            FROM df1, df2 INNER JOIN df3 ON df2.a = df3.a
            WHERE df1.a = df2.a
            ORDER BY df1.a
        """,
        compare_with="sqlite",
        expected={
            "a": [1, 2, 3, 4, 5],
            "b": [1, 3, 4, 4, 5],
            "c": [1, 3, 4, 5, 7],
        },
    )

    # OR filter on already-joined tables (residual, not join predicate)
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT df1.a, df1.b, df3.c
            FROM df1, df2, df3
            WHERE df1.a = df2.a
              AND df1.a = df3.a
              AND (df1.b = df2.b OR df1.b = df3.b)
            ORDER BY df1.a
        """,
        compare_with="sqlite",
        expected={"a": [2, 3], "b": [3, 4], "c": [3, 4]},
    )


def test_implicit_cross_join() -> None:
    t1 = pl.DataFrame({"x": [1, 2, 3]})
    t2 = pl.DataFrame({"y": [10, 20]})
    frames: dict[str, pl.DataFrame | pl.LazyFrame] = {"t1": t1, "t2": t2}

    # no join predicate → cross join
    assert_sql_matches(
        frames=frames,
        query="SELECT * FROM t1, t2 ORDER BY x, y",
        compare_with="sqlite",
        expected={"x": [1, 1, 2, 2, 3, 3], "y": [10, 20, 10, 20, 10, 20]},
    )

    # single-table filter → cross join + filter
    assert_sql_matches(
        frames=frames,
        query="SELECT * FROM t1, t2 WHERE t1.x > 1 ORDER BY t1.x, t2.y",
        compare_with="sqlite",
        expected={"x": [2, 2, 3, 3], "y": [10, 20, 10, 20]},
    )


def test_implicit_self_join() -> None:
    t = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

    assert_sql_matches(
        frames={"t": t},
        query="""
            SELECT a.id, a.val AS val_a, b.val AS val_b
            FROM t AS a, t AS b
            WHERE a.id = b.id
            ORDER BY a.id
        """,
        compare_with="sqlite",
        expected={"id": [1, 2, 3], "val_a": [10, 20, 30], "val_b": [10, 20, 30]},
    )


def test_non_equi_join() -> None:
    orders = pl.DataFrame(
        {"id": [1, 2, 3, 4, 5], "amount": [50, 100, 200, 150, 300]},
    )
    thresholds = pl.DataFrame(
        {"tier": ["bronze", "silver", "gold"], "min_amount": [0, 100, 200]}
    )
    ranges = pl.DataFrame(
        {"label": ["low", "mid", "high"], "lo": [0, 100, 200], "hi": [99, 199, 500]},
    )
    frames: dict[str, pl.DataFrame | pl.LazyFrame] = {
        "orders": orders,
        "thresholds": thresholds,
    }

    # strict greater-than
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT orders.id, orders.amount, thresholds.tier
            FROM orders
            INNER JOIN thresholds ON orders.amount > thresholds.min_amount
            ORDER BY orders.id, thresholds.min_amount
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 3, 4, 4, 5, 5, 5],
            "amount": [50, 100, 200, 200, 150, 150, 300, 300, 300],
            "tier": [
                "bronze",
                "bronze",
                "bronze",
                "silver",
                "bronze",
                "silver",
                "bronze",
                "silver",
                "gold",
            ],
        },
    )

    # not-equal
    t1 = pl.DataFrame({"x": [1, 2, 3]})
    t2 = pl.DataFrame({"y": [1, 2, 3]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="""
            SELECT t1.x, t2.y
            FROM t1 INNER JOIN t2 ON t1.x != t2.y
            ORDER BY t1.x, t2.y
        """,
        compare_with="sqlite",
        expected={"x": [1, 1, 2, 2, 3, 3], "y": [2, 3, 1, 3, 1, 2]},
    )

    expected: dict[str, Sequence[Any]] = {
        "id": [1, 2, 3, 4, 5],
        "amount": [50, 100, 200, 150, 300],
        "label": ["low", "mid", "high", "mid", "high"],
    }

    # explicit JOIN ... ON
    assert_sql_matches(
        frames={"orders": orders, "ranges": ranges},
        query="""
            SELECT orders.id, orders.amount, ranges.label
            FROM orders
            INNER JOIN ranges ON
                (orders.amount >= ranges.lo) AND
                (orders.amount <= ranges.hi)
            ORDER BY orders.id
        """,
        compare_with="sqlite",
        expected=expected,
    )

    # implicit join with the same condition
    assert_sql_matches(
        frames={"orders": orders, "ranges": ranges},
        query="""
            SELECT orders.id, orders.amount, ranges.label
            FROM orders, ranges
            WHERE orders.amount >= ranges.lo AND orders.amount <= ranges.hi
            ORDER BY orders.id
        """,
        compare_with="sqlite",
        expected=expected,
    )


def test_mixed_equi_non_equi_join() -> None:
    products = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": ["A", "A", "B", "B"],
            "price": [10, 20, 30, 40],
        }
    )
    discounts = pl.DataFrame(
        {
            "category": ["A", "A", "B"],
            "min_price": [0, 15, 25],
            "discount": [5, 10, 15],
        }
    )

    assert_sql_matches(
        frames={"products": products, "discounts": discounts},
        query="""
            SELECT products.id, products.category, products.price, discounts.discount
            FROM products
            INNER JOIN discounts
              ON products.category = discounts.category
             AND products.price > discounts.min_price
            ORDER BY products.id, discounts.min_price
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 2, 3, 4],
            "category": ["A", "A", "A", "B", "B"],
            "price": [10, 20, 20, 30, 40],
            "discount": [5, 5, 10, 15, 15],
        },
    )


def test_non_equi_self_join_errors() -> None:
    tbl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 3, 2]})

    # qualified self-join non-equi condition: both sides reference same table
    with pytest.raises(SQLInterfaceError, match="references both"):
        pl.sql("SELECT * FROM tbl LEFT JOIN tbl ON tbl.a != tbl.b")

    # unqualified self-join non-equi condition: ambiguous column reference
    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("SELECT * FROM tbl LEFT JOIN tbl ON a >= b")


def test_join_error_cases() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 3, 4, 4, 5]})
    df2 = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0, 3, 4, 5, 6]})
    df3 = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [0, 3, 4, 5, 7], "c": [1, 3, 4, 5, 7]}
    )

    # OR-connected cross-table predicates in implicit join
    with pytest.raises(SQLInterfaceError, match="cross-table predicates"):
        pl.sql("SELECT * FROM df1, df2 WHERE df1.a = df2.a OR df1.b = df2.b")

    # OR cross-table predicate for third table
    with pytest.raises(SQLInterfaceError, match="cross-table predicates"):
        pl.sql(
            """
            SELECT * FROM df1, df2, df3
            WHERE df1.a = df2.a
              AND (df3.a = df1.a OR df3.b = df2.b)
            """
        )

    # unnamed derived table in implicit join
    with pytest.raises(SQLInterfaceError, match="provide an alias"):
        pl.sql("SELECT * FROM (SELECT 1 AS x), df2")

    # both sides of ON reference the same table
    with pytest.raises(
        SQLInterfaceError,
        match="both sides of the ON predicate must reference different tables",
    ):
        pl.sql("SELECT * FROM df1 INNER JOIN df2 ON df2.a = df2.b")

    # ambiguous unqualified column in ON
    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("SELECT * FROM df1 INNER JOIN df2 ON a = a")

    # OR in explicit JOIN ON clause
    with pytest.raises(SQLInterfaceError, match="OR"):
        pl.sql("SELECT * FROM df1 INNER JOIN df2 ON df1.a = df2.a OR df1.b = df2.b")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # INNER joins
        (
            "SELECT df1.* FROM df1 INNER JOIN df2 USING (a)",
            {"a": [1, 3], "b": ["x", "z"], "c": [100, 300]},
        ),
        (
            "SELECT df2.* FROM df1 INNER JOIN df2 USING (a)",
            {"a": [1, 3], "b": ["qq", "pp"], "c": [400, 500]},
        ),
        (
            "SELECT df1.* FROM df2 INNER JOIN df1 USING (a)",
            {"a": [1, 3], "b": ["x", "z"], "c": [100, 300]},
        ),
        (
            "SELECT df2.* FROM df2 INNER JOIN df1 USING (a)",
            {"a": [1, 3], "b": ["qq", "pp"], "c": [400, 500]},
        ),
        # LEFT joins
        (
            "SELECT df1.* FROM df1 LEFT JOIN df2 USING (a)",
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]},
        ),
        (
            "SELECT df2.* FROM df1 LEFT JOIN df2 USING (a)",
            {"a": [1, 3, None], "b": ["qq", "pp", None], "c": [400, 500, None]},
        ),
        (
            "SELECT df1.* FROM df2 LEFT JOIN df1 USING (a)",
            {"a": [1, 3, None], "b": ["x", "z", None], "c": [100, 300, None]},
        ),
        (
            "SELECT df2.* FROM df2 LEFT JOIN df1 USING (a)",
            {"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]},
        ),
        # RIGHT joins
        (
            "SELECT df1.* FROM df1 RIGHT JOIN df2 USING (a)",
            {"a": [1, 3, None], "b": ["x", "z", None], "c": [100, 300, None]},
        ),
        (
            "SELECT df2.* FROM df1 RIGHT JOIN df2 USING (a)",
            {"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]},
        ),
        (
            "SELECT df1.* FROM df2 RIGHT JOIN df1 USING (a)",
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]},
        ),
        (
            "SELECT df2.* FROM df2 RIGHT JOIN df1 USING (a)",
            {"a": [1, 3, None], "b": ["qq", "pp", None], "c": [400, 500, None]},
        ),
        # FULL joins
        (
            "SELECT df1.* FROM df1 FULL JOIN df2 USING (a)",
            {
                "a": [1, 2, 3, None],
                "b": ["x", "y", "z", None],
                "c": [100, 200, 300, None],
            },
        ),
        (
            "SELECT df2.* FROM df1 FULL JOIN df2 USING (a)",
            {
                "a": [1, 3, 4, None],
                "b": ["qq", "pp", "oo", None],
                "c": [400, 500, 600, None],
            },
        ),
        (
            "SELECT df1.* FROM df2 FULL JOIN df1 USING (a)",
            {
                "a": [1, 2, 3, None],
                "b": ["x", "y", "z", None],
                "c": [100, 200, 300, None],
            },
        ),
        (
            "SELECT df2.* FROM df2 FULL JOIN df1 USING (a)",
            {
                "a": [1, 3, 4, None],
                "b": ["qq", "pp", "oo", None],
                "c": [400, 500, 600, None],
            },
        ),
    ],
)
def test_wildcard_resolution_and_join_order(
    query: str, expected: dict[str, Any]
) -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]})
    df2 = pl.DataFrame({"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]})

    res = pl.sql(query).collect()
    assert_frame_equal(
        res,
        pl.DataFrame(expected),
        check_row_order=False,
    )


def test_natural_joins_01() -> None:
    df1 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 4],
            "FirstName": ["Jernau Morat", "Cheradenine", "Byr", "Diziet"],
            "LastName": ["Gurgeh", "Zakalwe", "Genar-Hofoen", "Sma"],
        }
    )
    df2 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 5],
            "Role": ["Protagonist", "Protagonist", "Protagonist", "Antagonist"],
            "Book": [
                "Player of Games",
                "Use of Weapons",
                "Excession",
                "Consider Phlebas",
            ],
        }
    )
    df3 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 4],
            "Affiliation": ["Culture", "Culture", "Culture", "Shellworld"],
            "Species": ["Pan-human", "Human", "Human", "Oct"],
        }
    )
    df4 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 6],
            "Ship": [
                "Limiting Factor",
                "Xenophobe",
                "Grey Area",
                "Falling Outside The Normal Moral Constraints",
            ],
            "Drone": ["Flere-Imsaho", "Skaffen-Amtiskaw", "Eccentric", "Psychopath"],
        }
    )
    with pl.SQLContext(
        {"df1": df1, "df2": df2, "df3": df3, "df4": df4}, eager=True
    ) as ctx:
        res = ctx.execute(
            """
            SELECT *
            FROM df1
            NATURAL LEFT JOIN df2
            NATURAL INNER JOIN df3
            NATURAL LEFT JOIN df4
            ORDER BY ALL
            """
        )
        assert res.rows(named=True) == [
            {
                "CharacterID": 1,
                "FirstName": "Jernau Morat",
                "LastName": "Gurgeh",
                "Role": "Protagonist",
                "Book": "Player of Games",
                "Affiliation": "Culture",
                "Species": "Pan-human",
                "Ship": "Limiting Factor",
                "Drone": "Flere-Imsaho",
            },
            {
                "CharacterID": 2,
                "FirstName": "Cheradenine",
                "LastName": "Zakalwe",
                "Role": "Protagonist",
                "Book": "Use of Weapons",
                "Affiliation": "Culture",
                "Species": "Human",
                "Ship": "Xenophobe",
                "Drone": "Skaffen-Amtiskaw",
            },
            {
                "CharacterID": 3,
                "FirstName": "Byr",
                "LastName": "Genar-Hofoen",
                "Role": "Protagonist",
                "Book": "Excession",
                "Affiliation": "Culture",
                "Species": "Human",
                "Ship": "Grey Area",
                "Drone": "Eccentric",
            },
            {
                "CharacterID": 4,
                "FirstName": "Diziet",
                "LastName": "Sma",
                "Role": None,
                "Book": None,
                "Affiliation": "Shellworld",
                "Species": "Oct",
                "Ship": None,
                "Drone": None,
            },
        ]

    # misc errors
    with pytest.raises(SQLSyntaxError, match=r"did you mean COLUMNS\(\*\)\?"):
        pl.sql("SELECT * FROM df1 NATURAL JOIN df2 WHERE COLUMNS('*') >= 5")

    with pytest.raises(SQLSyntaxError, match=r"COLUMNS expects a regex"):
        pl.sql("SELECT COLUMNS(1234) FROM df1 NATURAL JOIN df2")


@pytest.mark.parametrize(
    ("cols_constraint", "expect_data"),
    [
        (">= 5", [(8, 8, 6)]),
        ("< 7", [(5, 4, 4)]),
        ("< 8", [(5, 4, 4), (7, 4, 4), (0, 7, 2)]),
        ("!= 4", [(8, 8, 6), (2, 8, 6), (0, 7, 2)]),
    ],
)
def test_natural_joins_02(cols_constraint: str, expect_data: list[tuple[int]]) -> None:
    df1 = pl.DataFrame(
        {
            "x": [1, 5, 3, 8, 6, 7, 4, 0, 2],
            "y": [3, 4, 6, 8, 3, 4, 1, 7, 8],
        }
    )
    df2 = pl.DataFrame(
        {
            "y": [0, 4, 0, 8, 0, 4, 0, 7, None],
            "z": [9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
    )
    actual = pl.sql(
        f"""
        SELECT *
        FROM df1 NATURAL JOIN df2
        WHERE COLUMNS(*) {cols_constraint}
        """
    ).collect()

    df_expected = pl.DataFrame(expect_data, schema=actual.columns, orient="row")
    assert_frame_equal(actual, df_expected, check_row_order=False)


@pytest.mark.parametrize(
    "join_clause",
    [
        """
        df2 JOIN df3 ON
        df2.CharacterID = df3.CharacterID
        """,
        """
        df2 INNER JOIN (
          df3 JOIN df4 ON df3.CharacterID = df4.CharacterID
        ) AS r0 ON df2.CharacterID = df3.CharacterID
        """,
    ],
)
def test_nested_join(join_clause: str) -> None:
    df1 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 4],
            "FirstName": ["Jernau Morat", "Cheradenine", "Byr", "Diziet"],
            "LastName": ["Gurgeh", "Zakalwe", "Genar-Hofoen", "Sma"],
        }
    )
    df2 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 5],
            "Role": ["Protagonist", "Protagonist", "Protagonist", "Antagonist"],
            "Book": [
                "Player of Games",
                "Use of Weapons",
                "Excession",
                "Consider Phlebas",
            ],
        }
    )
    df3 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 5, 6],
            "Affiliation": ["Culture", "Culture", "Culture", "Shellworld"],
            "Species": ["Pan-human", "Human", "Human", "Oct"],
        }
    )
    df4 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 6],
            "Ship": [
                "Limiting Factor",
                "Xenophobe",
                "Grey Area",
                "Falling Outside The Normal Moral Constraints",
            ],
            "Drone": ["Flere-Imsaho", "Skaffen-Amtiskaw", "Eccentric", "Psychopath"],
        }
    )

    with pl.SQLContext(
        {"df1": df1, "df2": df2, "df3": df3, "df4": df4}, eager=True
    ) as ctx:
        res = ctx.execute(
            f"""
            SELECT df1.CharacterID, df1.FirstName, df2.Role, df3.Species
            FROM df1
            INNER JOIN ({join_clause}) AS r99
            ON df1.CharacterID = df2.CharacterID
            ORDER BY ALL
            """
        )
        assert res.rows(named=True) == [
            {
                "CharacterID": 1,
                "FirstName": "Jernau Morat",
                "Role": "Protagonist",
                "Species": "Pan-human",
            },
            {
                "CharacterID": 2,
                "FirstName": "Cheradenine",
                "Role": "Protagonist",
                "Species": "Human",
            },
        ]


def test_miscellaneous_cte_join_aliasing() -> None:
    ctx = pl.SQLContext()
    res = ctx.execute(
        """
        WITH t AS (SELECT a FROM (VALUES(1),(2)) tbl(a))
        SELECT * FROM t CROSS JOIN t
        """,
        eager=True,
    )
    assert sorted(res.rows()) == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_nested_joins_17381() -> None:
    df = pl.DataFrame({"id": ["one", "two"]})

    ctx = pl.SQLContext({"a": df})
    res = ctx.execute(
        """
        -- the interaction of the (unused) CTE and the nested subquery resulted
        -- in arena mutation/cleanup that wasn't accounted for, affecting state
        WITH c AS (SELECT a.id FROM a)
        SELECT *
        FROM a
        WHERE id IN (
            SELECT a2.id
            FROM a
            INNER JOIN a AS a2 ON a.id = a2.id
        )
        """,
        eager=True,
    )
    assert set(res["id"]) == {"one", "two"}


def test_unnamed_nested_join_relation() -> None:
    df = pl.DataFrame({"a": 1})

    with (
        pl.SQLContext({"left": df, "right": df}) as ctx,
        pytest.raises(
            SQLInterfaceError,
            match="self-join requires table aliases",
        ),
    ):
        ctx.execute(
            """
            SELECT *
            FROM left
            JOIN (right JOIN right ON right.a = right.a)
            ON left.a = right.a
            """
        )


def test_nulls_equal_19624() -> None:
    df1 = pl.DataFrame({"a": [1, 2, None, None]})
    df2 = pl.DataFrame({"a": [1, 1, 2, 2, None], "b": [0, 1, 2, 3, 4]})

    # left join
    res_df = df1.join(df2, how="left", on="a", nulls_equal=False, validate="1:m")
    expected_df = pl.DataFrame(
        {"a": [1, 1, 2, 2, None, None], "b": [0, 1, 2, 3, None, None]}
    )
    assert_frame_equal(res_df, expected_df)
    res_df = df2.join(df1, how="left", on="a", nulls_equal=False, validate="m:1")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2, None], "b": [0, 1, 2, 3, 4]})
    assert_frame_equal(res_df, expected_df)

    # inner join
    res_df = df1.join(df2, how="inner", on="a", nulls_equal=False, validate="1:m")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3]})
    assert_frame_equal(res_df, expected_df)
    res_df = df2.join(df1, how="inner", on="a", nulls_equal=False, validate="m:1")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3]})
    assert_frame_equal(res_df, expected_df)


def test_join_on_literal_string_comparison() -> None:
    df1 = pl.DataFrame(
        {
            "name": ["alice", "bob", "adam", "charlie"],
            "role": ["admin", "user", "admin", "user"],
        }
    )
    df2 = pl.DataFrame(
        {
            "name": ["alice", "bob", "charlie", "adam"],
            "dept": ["IT", "HR", "IT", "SEC"],
        }
    )
    query = """
        SELECT df1.name, df1.role, df2.dept
        FROM df1
        INNER JOIN df2 ON df1.name = df2.name AND df1.role = 'admin'
        ORDER BY df1.name
    """
    df_expected = pl.DataFrame(
        data=[("adam", "admin", "SEC"), ("alice", "admin", "IT")],
        schema={"name": str, "role": str, "dept": str},
        orient="row",
    )
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("expression", "expected_length"),
    [
        ("LOWER(df1.text) = df2.text", 2),  # case conversion
        ("SUBSTR(df1.code, 1, 2) = SUBSTR(df2.code, 1, 2)", 3),  # first letter match
        ("LENGTH(df1.text) = LENGTH(df2.text)", 5),  # cartesian on matching lengths
    ],
)
def test_join_on_expression_conditions(expression: str, expected_length: int) -> None:
    df1 = pl.DataFrame(
        {
            "text": ["HELLO", "WORLD", "FOO"],
            "code": ["ABC", "DEF", "GHI"],
        }
    )
    df2 = pl.DataFrame(
        {
            "text": ["hello", "world", "bar"],
            "code": ["ABX", "DEY", "GHZ"],
        }
    )
    query = f"""
        SELECT df1.text AS text1, df2.text AS text2
        FROM df1
        INNER JOIN df2 ON {expression}
        ORDER BY text1
    """
    res = pl.sql(query, eager=True)
    assert len(res) == expected_length


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "select_cols", "expected", "schema"),
    [
        (
            pl.DataFrame(
                {
                    "category": ["fruit", "fruit", "vegetable"],
                    "name": ["apple", "banana", "carrot"],
                    "code": [1, 2, 3],
                }
            ),
            pl.DataFrame(
                {
                    "category": ["fruit", "fruit", "vegetable"],
                    "type": ["sweet", "tropical", "root"],
                    "code_doubled": [2, 4, 6],
                }
            ),
            "df1.category = df2.category AND (df1.code * 2) = df2.code_doubled",
            "df1.name, df1.code, df2.type",
            [("apple", 1, "sweet"), ("banana", 2, "tropical"), ("carrot", 3, "root")],
            ["name", "code", "type"],
        ),
        (
            pl.DataFrame({"id": [1, 2, 3], "name": ["ALICE", "BOB", "CHARLIE"]}),
            pl.DataFrame({"id": [1, 2, 3], "match": ["alice", "bob", "charlie"]}),
            "df1.id = df2.id AND LOWER(df1.name) = df2.match",
            "df1.id, df1.name, df2.match",
            [(1, "ALICE", "alice"), (2, "BOB", "bob"), (3, "CHARLIE", "charlie")],
            ["id", "name", "match"],
        ),
        (
            pl.DataFrame({"x": [2, 4, 6], "y": [1, 2, 3]}),
            pl.DataFrame({"a": [4, 8, 12], "b": [1, 2, 3]}),
            "df1.x * 2 = df2.a AND df1.y = df2.b",
            "df1.x, df1.y, df2.a",
            [(2, 1, 4), (4, 2, 8), (6, 3, 12)],
            ["x", "y", "a"],
        ),
    ],
)
def test_join_on_mixed_expression_conditions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    select_cols: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    query = f"""
        SELECT {select_cols}
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY ALL
    """
    df_expected = pl.DataFrame(expected, schema=schema, orient="row")
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "expected"),
    [
        (
            pl.DataFrame({"text": ["  Hello  ", "  World  ", "  Test  "]}),
            pl.DataFrame({"text": ["hello", "world", "other"]}),
            "LOWER(TRIM(df1.text)) = df2.text",
            [("  Hello  ", "hello"), ("  World  ", "world")],
        ),
        (
            pl.DataFrame({"code": ["PREFIX_A", "SECOND_B", "OTHERS_C"]}),
            pl.DataFrame({"code": ["prefix", "second", "others"]}),
            "LOWER(SUBSTR(df1.code,1,6)) = df2.code",
            [("OTHERS_C", "others"), ("PREFIX_A", "prefix"), ("SECOND_B", "second")],
        ),
        (
            pl.DataFrame({"name": ["abc", "abcde", "x"]}),
            pl.DataFrame({"len": [3, 5, 1]}),
            "LENGTH(df1.name) = df2.len",
            [("x", 1), ("abc", 3), ("abcde", 5)],
        ),
    ],
)
def test_join_on_nested_function_expressions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    expected: list[tuple[Any, ...]],
) -> None:
    col1 = df1.columns[0]
    col2 = df2.columns[0]

    query = f"""
        SELECT df1.{col1} AS col1, df2.{col2} AS col2
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY df2.{col2}
    """
    df_expected = pl.DataFrame(expected, schema=["col1", "col2"], orient="row")
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "select_cols", "expected", "schema"),
    [
        (
            pl.DataFrame(
                {"id": [1, 2, 3], "category": ["A", "B", "A"], "multiplier": [2, 3, 4]}
            ),
            pl.DataFrame(
                {"id": [1, 2, 3], "base": [5, 15, 20], "category": ["A", "B", "C"]}
            ),
            "df1.id = df2.id AND df1.multiplier * 5 = df2.base AND df1.category = 'A'",
            "df1.id, df1.multiplier, df2.base",
            [(3, 4, 20)],
            ["id", "multiplier", "base"],
        ),
        (
            pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
            pl.DataFrame({"id": [1, 2, 3], "target": [20, 40, 60]}),
            "df1.id = df2.id AND (df1.value * 2) = df2.target AND df1.id = 2",
            "df1.id, df1.value, df2.target",
            [(2, 20, 40)],
            ["id", "value", "target"],
        ),
        (
            pl.DataFrame(
                {
                    "x": [1, 2, 3],
                    "type": ["A", "B", "A"],
                    "status": ["active", "inactive", "active"],
                }
            ),
            pl.DataFrame({"x": [1, 2, 3], "data": ["foo", "bar", "baz"]}),
            "df1.x = df2.x AND df1.type = 'A' AND df1.status = 'active'",
            "df1.x, df2.data",
            [(1, "foo"), (3, "baz")],
            ["x", "data"],
        ),
    ],
)
def test_join_on_expression_with_literals(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    select_cols: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    query = f"""
        SELECT {select_cols}
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY ALL
    """
    df_expected = pl.DataFrame(
        expected,
        schema=schema,
        orient="row",
    )
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "reversed_join_constraint", "expected", "schema"),
    [
        (
            pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]}),
            pl.DataFrame({"id": [2, 3, 4], "val": ["x", "y", "z"]}),
            "df1.id = df2.id",
            "df2.id = df1.id",
            [(2, "b", "x"), (3, "c", "y")],
            ["id", "val1", "val2"],
        ),
        (
            pl.DataFrame({"x": [1, 2, 3]}),
            pl.DataFrame({"y": [2, 4, 6]}),
            "df1.x * 2 = df2.y",
            "df2.y = (df1.x * 2)",
            [(1, 2), (2, 4), (3, 6)],
            ["x", "y"],
        ),
        (
            pl.DataFrame({"a": [5, 10, 15]}),
            pl.DataFrame({"b": [10, 20, 30]}),
            "(df1.a + df1.a) = df2.b",
            "df2.b = (df1.a + df1.a)",
            [(5, 10), (10, 20), (15, 30)],
            ["a", "b"],
        ),
    ],
)
def test_join_on_reversed_constraint_order(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    reversed_join_constraint: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    select_cols = (
        "df1.id, df1.val AS val1, df2.val AS val2"
        if len(schema) == 3
        else ", ".join(f"df{i + 1}.{col}" for i, col in enumerate(schema))
    )
    df_expected = pl.DataFrame(
        expected,
        schema=schema,
        orient="row",
    )
    for constraint in (join_constraint, reversed_join_constraint):
        res = pl.sql(
            query=f"""
                SELECT {select_cols}
                FROM df1
                INNER JOIN df2 ON {constraint}
                ORDER BY ALL
            """,
            eager=True,
        )
        assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "expected", "schema"),
    [
        (
            pl.DataFrame({"a": [1, 2, 3]}),
            pl.DataFrame({"b": [2, 4, 6]}),
            "a * 2 = b",
            [(1, 2), (2, 4), (3, 6)],
            ["a", "b"],
        ),
        (
            pl.DataFrame({"x": [5, 10, 15], "y": [3, 5, 7]}),
            pl.DataFrame({"sum": [8, 15, 22]}),
            "x + y = sum",
            [(5, 3, 8), (10, 5, 15), (15, 7, 22)],
            ["x", "y", "sum"],
        ),
        (
            pl.DataFrame({"name": ["abc", "hello", "test"]}),
            pl.DataFrame({"len": [3, 5, 4]}),
            "LENGTH(name) = len",
            [("abc", 3), ("hello", 5), ("test", 4)],
            ["name", "len"],
        ),
    ],
)
def test_join_on_unqualified_expressions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    df1_cols = ", ".join(f"df1.{col}" for col in df1.columns)
    df2_cols = ", ".join(f"df2.{col}" for col in df2.columns)

    query = f"""
        SELECT {df1_cols}, {df2_cols}
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY ALL
    """
    df_expected = pl.DataFrame(
        expected,
        schema=schema,
        orient="row",
    )
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


def test_multiway_join_chain_with_aliased_cols() -> None:
    # tracking/resolving constraints for 3-way (or more) joins can be... "fun" :)
    # ref: https://github.com/pola-rs/polars/issues/25126

    df1 = pl.DataFrame({"a": [111, 222], "x1": ["df1", "df1"]})
    df2 = pl.DataFrame({"a": [333, 111], "b": [444, 222], "x2": ["df2", "df2"]})
    df3 = pl.DataFrame({"a": [222, 111], "x3": ["df3", "df3"]})

    for query, expected_cols, expected_row in (
        (
            # three-way join where "a" exists in all three frames (df1, df2, df3)
            """
            SELECT * FROM df3
            INNER JOIN df2 ON df2.b = df3.a
            INNER JOIN df1 ON df1.a = df2.a
            """,
            ["a", "x3", "a:df2", "b", "x2", "a:df1", "x1"],
            (222, "df3", 111, 222, "df2", 111, "df1"),
        ),
        (
            # almost the same, but the final constraint on "a" refers back to df1
            """
            SELECT * FROM df3
            INNER JOIN df2 ON df2.b = df3.a
            INNER JOIN df1 ON df1.a = df3.a
            """,
            ["a", "x3", "a:df2", "b", "x2", "a:df1", "x1"],
            (222, "df3", 111, 222, "df2", 222, "df1"),
        ),
    ):
        res = pl.sql(query, eager=True)

        assert res.height == 1
        assert res.columns == expected_cols
        assert res.row(0) == expected_row


@pytest.mark.parametrize(
    ("join_condition", "expected_error"),
    [
        (
            "(df1.id + df2.val) = df2.id",
            r"unsupported join condition: left side references both 'df1' and 'df2'",
        ),
        (
            "df1.id = (df2.id + df1.val)",
            r"unsupported join condition: right side references both 'df1' and 'df2'",
        ),
    ],
)
def test_unsupported_join_conditions(join_condition: str, expected_error: str) -> None:
    # note: this is technically valid (if unusual) SQL, but we don't support it
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "val": [20, 30, 40]})

    with pytest.raises(SQLInterfaceError, match=expected_error):
        pl.sql(f"SELECT * FROM df1 INNER JOIN df2 ON {join_condition}")


def test_ambiguous_column_detection_in_joins() -> None:
    # unqualified column references that exist in multiple tables should raise
    # an error (with a helpful suggestion about qualifying the reference)
    with pytest.raises(
        SQLInterfaceError,
        match=r'ambiguous reference to column "k" \(use one of: a\.k, c\.k\)',
    ):
        pl.sql(
            query="""
                WITH
                  a AS (SELECT 0 AS k),
                  c AS (SELECT 0 AS k)
                SELECT k FROM a JOIN c ON a.k = c.k
            """,
            eager=True,
        )


def test_duplicate_column_detection_via_wildcard() -> None:
    # selecting a column explicitly that is already included in a qualified
    # wildcard from the same table should raise a duplicate column error
    a = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
    b = pl.DataFrame({"id": [1, 2], "y": [30, 40]})

    with pytest.raises(
        SQLInterfaceError,
        match=r"column 'id' is duplicated in the SELECT",
    ):
        pl.sql("SELECT a.*, a.id FROM a JOIN b ON a.id = b.id", eager=True)


def test_qualified_wildcard_multiway_join() -> None:
    df1 = pl.DataFrame({"id": [1, 2], "a": ["x", "y"]})
    df2 = pl.DataFrame({"id": [1, 2], "b": ["p", "q"]})
    df3 = pl.DataFrame({"id": [1, 2], "c": ["m", "n"]})

    res = pl.sql("""
        SELECT df1.*, df2.*, df3.*
        FROM df1
        INNER JOIN df2 ON df1.id = df2.id
        INNER JOIN df3 ON df1.id = df3.id
        ORDER BY id
    """).collect()
    expected = pl.DataFrame(
        {
            "id": [1, 2],
            "a": ["x", "y"],
            "id:df2": [1, 2],
            "b": ["p", "q"],
            "id:df3": [1, 2],
            "c": ["m", "n"],
        }
    )
    assert_frame_equal(res, expected)


def test_qualified_wildcard_self_join() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "parent": [None, 1, 1],
            "name": ["root", "child1", "child2"],
        }
    )
    res = pl.sql("""
        SELECT child.*, parent.*
        FROM df AS child
        LEFT JOIN df AS parent ON child.parent = parent.id
        ORDER BY id
    """).collect()

    expected = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "parent": [None, 1, 1],
            "name": ["root", "child1", "child2"],
            "id:parent": [None, 1, 1],
            "parent:parent": [None, None, None],
            "name:parent": [None, "root", "root"],
        },
        schema_overrides={"parent:parent": pl.Int64},
    )
    assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    ("join_type", "result"),
    [
        (
            "INNER",
            {"k": [1], "v": ["a"], "k:df2": [1], "v:df2": ["x"]},
        ),
        (
            "LEFT",
            {"k": [1, 2], "v": ["a", "b"], "k:df2": [1, None], "v:df2": ["x", None]},
        ),
        (
            "RIGHT",
            {"k": [1, None], "v": ["a", None], "k:df2": [1, 3], "v:df2": ["x", "y"]},
        ),
    ],
)
def test_qualified_wildcard_join_types(join_type: str, result: dict[str, Any]) -> None:
    df1 = pl.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    df2 = pl.DataFrame({"k": [1, 3], "v": ["x", "y"]})

    actual = pl.sql(
        query=f"""
        SELECT df1.*, df2.*
        FROM df1 {join_type} JOIN df2 ON df1.k = df2.k
        """,
        eager=True,
    )
    expected = pl.DataFrame(result)
    assert_frame_equal(
        left=expected,
        right=actual,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (  # specific column conflicts with wildcard
            "SELECT a.id, b.* FROM a JOIN b ON a.id = b.id",
            {"id": [1, 2], "id:b": [1, 2], "y": [30, 40]},
        ),
        (  # specific column doesn't conflict with wildcard
            "SELECT b.y, a.* FROM a JOIN b ON a.id = b.id",
            {"y": [30, 40], "id": [1, 2], "x": [10, 20]},
        ),
        (  # single-table wildcard (no conflict, uses original names)
            "SELECT b.* FROM a JOIN b ON a.id = b.id",
            {"id": [1, 2], "y": [30, 40]},
        ),
        (  # table aliases (disambiguation should use the alias)
            "SELECT t1.*, t2.* FROM a AS t1 JOIN b AS t2 ON t1.id = t2.id",
            {"id": [1, 2], "x": [10, 20], "id:t2": [1, 2], "y": [30, 40]},
        ),
        (  # no column overlap (expect no disambiguation)
            "SELECT a.*, c.* FROM a JOIN c ON a.id = c.k",
            {"id": [1, 2], "x": [10, 20], "k": [1, 2], "z": [50, 60]},
        ),
        (  # reverse wildcard order (disambiguation follows *table* order)
            "SELECT b.*, a.* FROM a JOIN b ON a.id = b.id",
            {"id:b": [1, 2], "y": [30, 40], "id": [1, 2], "x": [10, 20]},
        ),
    ],
)
def test_qualified_wildcard_combinations(query: str, expected: dict[str, Any]) -> None:
    a = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
    b = pl.DataFrame({"id": [1, 2], "y": [30, 40]})
    c = pl.DataFrame({"k": [1, 2], "z": [50, 60]})

    assert_frame_equal(
        left=pl.DataFrame(expected),
        right=pl.sql(query).collect(),
        check_row_order=False,
    )


def test_natural_join_no_common_columns() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})

    with pytest.raises(SQLInterfaceError, match=r"no common columns.*'df1'.*'df2'"):
        pl.sql("SELECT * FROM df1 NATURAL JOIN df2").collect()


def test_cross_join_with_constraint_error() -> None:
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"b": [2]})

    # SQL parser rejects CROSS JOIN with ON/USING at the syntax level
    with pytest.raises(SQLInterfaceError, match="sql parser error"):
        pl.sql("SELECT * FROM df1 CROSS JOIN df2 ON df1.a = df2.b").collect()

    with pytest.raises(SQLInterfaceError, match="sql parser error"):
        pl.sql("SELECT * FROM df1 CROSS JOIN df2 USING (a)").collect()


def test_non_equi_join_inner_vs_left() -> None:
    """Non-equi INNER JOIN works via join_where; LEFT JOIN errors."""
    orders = pl.DataFrame({"id": [1, 2, 3], "amount": [50, 150, 300]})
    thresholds = pl.DataFrame({"tier": ["bronze", "gold"], "min_amount": [0, 200]})

    # INNER JOIN with non-equi condition succeeds
    assert_sql_matches(
        frames={"orders": orders, "thresholds": thresholds},
        query="""
            SELECT orders.id, orders.amount, thresholds.tier
            FROM orders
            INNER JOIN thresholds ON orders.amount > thresholds.min_amount
            ORDER BY orders.id, thresholds.min_amount
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 3],
            "amount": [50, 150, 300, 300],
            "tier": ["bronze", "bronze", "bronze", "gold"],
        },
    )

    # LEFT JOIN with non-equi condition errors (join_where doesn't support outer)
    with pytest.raises(SQLInterfaceError, match="non-equi join conditions"):
        pl.SQLContext(orders=orders, thresholds=thresholds).execute(
            """
            SELECT orders.id, orders.amount, thresholds.tier
            FROM orders
            LEFT JOIN thresholds ON orders.amount > thresholds.min_amount
            ORDER BY orders.id, thresholds.min_amount
            """
        ).collect()


def test_non_equi_self_join_with_aliases() -> None:
    tbl = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

    assert_sql_matches(
        frames={"tbl": tbl},
        query="""
            SELECT a.id AS id_a, b.id AS id_b, a.val AS val_a, b.val AS val_b
            FROM tbl AS a INNER JOIN tbl AS b ON a.val > b.val
            ORDER BY a.id, b.id
        """,
        compare_with="sqlite",
        expected={
            "id_a": [2, 3, 3],
            "id_b": [1, 1, 2],
            "val_a": [20, 30, 30],
            "val_b": [10, 10, 20],
        },
    )


def test_implicit_join_four_tables() -> None:
    df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [1, 2, 3], "b": [100, 200, 300]})
    df3 = pl.DataFrame({"id": [1, 2, 3], "c": ["x", "y", "z"]})
    df4 = pl.DataFrame({"id": [1, 2, 3], "d": [1.1, 2.2, 3.3]})

    assert_sql_matches(
        frames={"df1": df1, "df2": df2, "df3": df3, "df4": df4},
        query="""
            SELECT df1.id, df1.a, df2.b, df3.c, df4.d
            FROM df1, df2, df3, df4
            WHERE df1.id = df2.id AND df2.id = df3.id AND df3.id = df4.id
            ORDER BY df1.id
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3],
            "a": [10, 20, 30],
            "b": [100, 200, 300],
            "c": ["x", "y", "z"],
            "d": [1.1, 2.2, 3.3],
        },
    )


def test_mixed_equi_non_equi_left_join() -> None:
    """Non-equi conditions with outer joins should raise an error.

    The join_where path (Cross + Filter) does not preserve outer join semantics,
    so we error rather than silently producing wrong results. When outer join
    support is added to join_where, this test should be updated to validate
    the correct result instead.
    """
    products = pl.DataFrame(
        {"id": [1, 2, 3], "cat": ["A", "B", "C"], "price": [10, 30, 50]}
    )
    discounts = pl.DataFrame({"cat": ["A", "B"], "min_price": [5, 25], "disc": [1, 2]})

    with pytest.raises(SQLInterfaceError, match="non-equi join conditions"):
        pl.SQLContext(products=products, discounts=discounts).execute(
            """
            SELECT products.id, products.cat, discounts.disc
            FROM products
            LEFT JOIN discounts
              ON products.cat = discounts.cat
              AND products.price > discounts.min_price
            ORDER BY products.id
            """
        ).collect()


def test_natural_right_and_full_join() -> None:
    """NATURAL RIGHT/FULL JOIN preserves unmatched rows appropriately."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})

    # RIGHT: unmatched right rows get NULL left columns
    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT id, a, b
            FROM df1 NATURAL RIGHT JOIN df2
            ORDER BY id
        """,
        compare_with="sqlite",
        expected={
            "id": [2, 3, 4],
            "a": [20, 30, None],
            "b": [200, 300, 400],
        },
    )

    # FULL: unmatched rows from both sides preserved
    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT id, a, b
            FROM df1 NATURAL FULL JOIN df2
            ORDER BY id
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4],
            "a": [10, 20, 30, None],
            "b": [None, 200, 300, 400],
        },
    )


def test_implicit_join_unqualified_filter() -> None:
    """Unqualified single-table predicates remain as residual filters."""
    t1 = pl.DataFrame({"x": [1, 2], "y": [10, 20]})
    t2 = pl.DataFrame({"z": [100, 200]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="SELECT * FROM t1, t2 WHERE x = 1 ORDER BY z",
        compare_with="sqlite",
        expected={"x": [1, 1], "y": [10, 10], "z": [100, 200]},
    )


def test_implicit_join_third_table_references_second() -> None:
    """Third table in an implicit join referencing only the second table."""
    t1 = pl.DataFrame({"a": [1, 2, 3]})
    t2 = pl.DataFrame({"b": [1, 2, 3], "c": [10, 20, 30]})
    t3 = pl.DataFrame({"c": [10, 20, 30], "d": ["x", "y", "z"]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3},
        query="""
            SELECT t1.a, t2.b, t3.d
            FROM t1, t2, t3
            WHERE t1.a = t2.b AND t2.c = t3.c
            ORDER BY t1.a
        """,
        compare_with="sqlite",
        expected={"a": [1, 2, 3], "b": [1, 2, 3], "d": ["x", "y", "z"]},
    )


def test_implicit_join_single_non_equi() -> None:
    """Implicit join with a single non-equi (greater-than) condition."""
    t1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    t2 = pl.DataFrame({"id": [1, 2, 3], "val": [15, 25, 35]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="""
            SELECT t1.id AS id1, t2.id AS id2, t1.val AS v1, t2.val AS v2
            FROM t1, t2
            WHERE t1.val > t2.val
            ORDER BY t1.id, t2.id
        """,
        compare_with="sqlite",
        expected={
            "id1": [2, 3, 3],
            "id2": [1, 1, 2],
            "v1": [20, 30, 30],
            "v2": [15, 15, 25],
        },
    )


def test_implicit_join_three_tables_overlapping_columns() -> None:
    """Three-table implicit join where all tables share the same column name.

    Regression test: ensures that when columns conflict across multiple
    implicit joins (requiring suffixing), the predicates are still resolved
    correctly against the right table.
    """
    t1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    t2 = pl.DataFrame({"id": [1, 2, 3], "val": [100, 200, 300]})
    t3 = pl.DataFrame({"id": [1, 2, 3], "val": [1000, 2000, 3000]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3},
        query="""
            SELECT t1.id, t1.val AS v1, t2.val AS v2, t3.val AS v3
            FROM t1, t2, t3
            WHERE t1.id = t2.id AND t2.id = t3.id
            ORDER BY t1.id
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3],
            "v1": [10, 20, 30],
            "v2": [100, 200, 300],
            "v3": [1000, 2000, 3000],
        },
    )


def test_implicit_join_aliased_derived_tables() -> None:
    """Implicit join between aliased derived tables (subqueries)."""
    assert_sql_matches(
        frames={
            "src": pl.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]}),
        },
        query="""
            SELECT a.id, a.x AS ax, b.x AS bx
            FROM (SELECT * FROM src WHERE id <= 2) AS a,
                 (SELECT * FROM src WHERE id >= 2) AS b
            WHERE a.id = b.id
        """,
        compare_with="sqlite",
        expected={"id": [2], "ax": [20], "bx": [20]},
    )


def test_implicit_join_mixed_with_explicit_multi_table() -> None:
    """Mixed implicit + explicit join involving three tables.

    Regression test: the implicit join predicate for the third table must
    be correctly extracted from the WHERE clause even when the first two
    tables are joined explicitly.
    """
    t1 = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    t2 = pl.DataFrame({"a": [1, 2, 3], "c": ["x", "y", "z"]})
    t3 = pl.DataFrame({"a": [1, 2, 3], "d": [100, 200, 300]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3},
        query="""
            SELECT t1.a, t1.b, t2.c, t3.d
            FROM t1 INNER JOIN t2 ON t1.a = t2.a, t3
            WHERE t1.a = t3.a
            ORDER BY t1.a
        """,
        compare_with="sqlite",
        expected={
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": ["x", "y", "z"],
            "d": [100, 200, 300],
        },
    )


def test_natural_semi_anti_join() -> None:
    """NATURAL JOIN combined with SEMI/ANTI should use common columns."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})

    ctx = pl.SQLContext({"df1": df1, "df2": df2}, eager=True)

    # NATURAL SEMI: keep left rows that match on common columns
    result = ctx.execute("""
        SELECT * FROM df1 NATURAL LEFT SEMI JOIN df2 ORDER BY id
    """)
    expected = pl.DataFrame({"id": [2, 3], "a": [20, 30]})
    assert_frame_equal(result, expected)

    # NATURAL ANTI: keep left rows that do NOT match on common columns
    result = ctx.execute("""
        SELECT * FROM df1 NATURAL LEFT ANTI JOIN df2 ORDER BY id
    """)
    expected = pl.DataFrame({"id": [1], "a": [10]})
    assert_frame_equal(result, expected)


def test_implicit_join_no_predicate_for_table_error() -> None:
    """Implicit join with OR-connected cross-table predicates must error.

    Regression test: when cross-table predicates cannot be extracted as
    join conditions, an explicit error should be raised rather than
    silently producing a cartesian product.
    """
    t1 = pl.DataFrame({"x": [1, 2]})
    t2 = pl.DataFrame({"y": [1, 2]})
    t3 = pl.DataFrame({"z": [1, 2]})

    # OR-connecting predicates for the third table should error
    with pytest.raises(SQLInterfaceError, match="cross-table predicates"):
        pl.sql("""
            SELECT * FROM t1, t2, t3
            WHERE t1.x = t2.y AND (t3.z = t1.x OR t3.z = t2.y)
        """)


def test_implicit_join_unqualified_columns() -> None:
    """Unqualified columns uniquely belonging to one table should resolve.

    Resolved via schema membership and used as join predicates (producing
    an INNER JOIN instead of a cartesian product).
    """
    t1 = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    t2 = pl.DataFrame({"c": [2, 3, 4], "d": [200, 300, 400]})

    # 'a' is unique to t1, 'c' is unique to t2 → should produce INNER JOIN
    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="SELECT a, b, c, d FROM t1, t2 WHERE a = c ORDER BY a",
        compare_with="sqlite",
        expected={"a": [2, 3], "b": [20, 30], "c": [2, 3], "d": [200, 300]},
    )


def test_implicit_join_unqualified_non_equi() -> None:
    """Unqualified columns with non-equi operators should also work."""
    t1 = pl.DataFrame({"x": [1, 2, 3]})
    t2 = pl.DataFrame({"y": [2, 3, 4]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="SELECT x, y FROM t1, t2 WHERE x < y ORDER BY x, y",
        compare_with="sqlite",
        expected={
            "x": [1, 1, 1, 2, 2, 3],
            "y": [2, 3, 4, 3, 4, 4],
        },
    )


def test_implicit_join_unqualified_mixed_with_filter() -> None:
    """Unqualified cross-table predicate AND single-table filter."""
    t1 = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    t2 = pl.DataFrame({"c": [1, 2, 3], "d": [100, 200, 300]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="SELECT a, d FROM t1, t2 WHERE a = c AND b > 10 ORDER BY a",
        compare_with="sqlite",
        expected={"a": [2, 3], "d": [200, 300]},
    )


def test_implicit_join_unqualified_three_tables() -> None:
    """Unqualified columns across three implicit tables."""
    t1 = pl.DataFrame({"a": [1, 2, 3], "v1": [10, 20, 30]})
    t2 = pl.DataFrame({"b": [1, 2, 3], "v2": [100, 200, 300]})
    t3 = pl.DataFrame({"c": [1, 2, 3], "v3": [1000, 2000, 3000]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3},
        query="""
            SELECT v1, v2, v3 FROM t1, t2, t3
            WHERE a = b AND b = c
            ORDER BY v1
        """,
        compare_with="sqlite",
        expected={
            "v1": [10, 20, 30],
            "v2": [100, 200, 300],
            "v3": [1000, 2000, 3000],
        },
    )


@pytest.mark.parametrize(
    "join_type",
    [
        "RIGHT",
        "FULL",
        "LEFT SEMI",
        "LEFT ANTI",
        "RIGHT SEMI",
        "RIGHT ANTI",
    ],
)
def test_non_equi_join_rejected_for_non_inner_types(join_type: str) -> None:
    """Non-equi conditions must error for all join types except INNER."""
    t1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    t2 = pl.DataFrame({"id": [1, 2, 3], "val": [15, 25, 35]})

    with pytest.raises(SQLInterfaceError, match="non-equi join conditions"):
        pl.sql(f"SELECT * FROM t1 {join_type} JOIN t2 ON t1.val > t2.val").collect()


def test_non_equi_join_both_tables_in_single_operand() -> None:
    """A single operand of a non-equi condition referencing both tables should error."""
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df2 = pl.DataFrame({"c": [1, 2, 3], "d": [10, 20, 30]})

    with pytest.raises(
        SQLInterfaceError, match="each side must reference a single table"
    ):
        pl.sql("SELECT * FROM df1 INNER JOIN df2 ON (df1.a + df2.c) > df2.d")


def test_non_equi_join_ambiguous_unqualified_column() -> None:
    """Unqualified column in non-equi that exists in both tables should error."""
    t1 = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    t2 = pl.DataFrame({"x": [4, 5, 6], "z": [40, 50, 60]})

    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("SELECT * FROM t1 INNER JOIN t2 ON x > t2.z")


def test_implicit_join_five_tables() -> None:
    """Five-table implicit join with chained predicates."""
    t1 = pl.DataFrame({"a": [1, 2], "v1": [10, 20]})
    t2 = pl.DataFrame({"b": [1, 2], "v2": [100, 200]})
    t3 = pl.DataFrame({"c": [1, 2], "v3": [1000, 2000]})
    t4 = pl.DataFrame({"d": [1, 2], "v4": [10000, 20000]})
    t5 = pl.DataFrame({"e": [1, 2], "v5": [100000, 200000]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
        query="""
            SELECT v1, v2, v3, v4, v5
            FROM t1, t2, t3, t4, t5
            WHERE a = b AND b = c AND c = d AND d = e
            ORDER BY v1
        """,
        compare_with="sqlite",
        expected={
            "v1": [10, 20],
            "v2": [100, 200],
            "v3": [1000, 2000],
            "v4": [10000, 20000],
            "v5": [100000, 200000],
        },
    )


def test_implicit_join_unqualified_ambiguous_column_error() -> None:
    """Unqualified column existing in both tables should not be extracted.

    Not extracted as a join predicate (ambiguous); should error or cross-join.
    """
    t1 = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    t2 = pl.DataFrame({"id": [1, 2], "b": [100, 200]})

    # 'id' exists in both tables — ambiguous. The cross-table detection
    # should catch that the unqualified 'id' spans both schemas.
    with pytest.raises(SQLInterfaceError, match=r"cross-table predicates|ambiguous"):
        pl.sql("""
            SELECT * FROM t1, t2 WHERE id = id
        """)


def test_ambiguous_unqualified_column_in_explicit_join_on() -> None:
    """Unqualified column in explicit JOIN ON that exists in both tables should error.

    Regression test for issue where Both was silently resolved instead of erroring.
    """
    t1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
    t2 = pl.DataFrame({"id": [1, 2, 3], "b": [100, 200, 300]})

    # 'id' exists in both tables — ambiguous when unqualified
    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("SELECT t1.a, t2.b FROM t1 INNER JOIN t2 ON id = t2.id")

    # reversed: qualified on left, ambiguous on right
    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("SELECT t1.a, t2.b FROM t1 INNER JOIN t2 ON t1.id = id")

    # both unqualified
    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("SELECT t1.a, t2.b FROM t1 INNER JOIN t2 ON id = id")


def test_cyclic_implicit_join_predicates() -> None:
    """Cyclic predicates (a=b, b=c, c=a) should produce correct results.

    All three predicates are logically redundant (c=a is implied by
    a=b AND b=c), but the query should still execute correctly.
    """
    t1 = pl.DataFrame({"a": [1, 2], "v1": [10, 20]})
    t2 = pl.DataFrame({"b": [1, 2], "v2": [100, 200]})
    t3 = pl.DataFrame({"c": [1, 2], "v3": [1000, 2000]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3},
        query="""
            SELECT v1, v2, v3 FROM t1, t2, t3
            WHERE a = b AND b = c AND c = a
            ORDER BY v1
        """,
        compare_with="sqlite",
        expected={
            "v1": [10, 20],
            "v2": [100, 200],
            "v3": [1000, 2000],
        },
    )


def test_non_equi_join_null_handling() -> None:
    """NULL values in non-equi join conditions should be excluded.

    Comparison operators return NULL (not true) when an operand is NULL,
    so rows with NULLs should not appear in the join result.
    """
    t1 = pl.DataFrame({"id": [1, 2, None], "val": [10, 20, None]})
    t2 = pl.DataFrame({"id": [1, 2, 3], "val": [15, 25, 35]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2},
        query="""
            SELECT t1.id AS id1, t2.id AS id2, t1.val AS v1, t2.val AS v2
            FROM t1 INNER JOIN t2 ON t1.val > t2.val
            ORDER BY id1, id2
        """,
        compare_with="sqlite",
        expected={
            "id1": [2],
            "id2": [1],
            "v1": [20],
            "v2": [15],
        },
    )


def test_full_outer_join_select_star_conflicting_columns() -> None:
    """FULL OUTER JOIN with SELECT * should suffix conflicting columns."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "val": [200, 300, 400]})

    res = pl.sql("""
        SELECT * FROM df1 FULL OUTER JOIN df2 ON df1.id = df2.id
        ORDER BY COALESCE(df1.id, df2.id)
    """).collect()

    assert res.columns == ["id", "val", "id:df2", "val:df2"]
    assert res.shape == (4, 4)
    # Unmatched left row (id=1)
    assert res.row(0) == (1, 10, None, None)
    # Matched rows
    assert res.row(1) == (2, 20, 2, 200)
    assert res.row(2) == (3, 30, 3, 300)
    # Unmatched right row (id=4)
    assert res.row(3) == (None, None, 4, 400)


def test_empty_using_clause_error() -> None:
    """Empty USING clause should be caught by the SQL parser."""
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"b": [2]})

    with pytest.raises((SQLInterfaceError, SQLSyntaxError)):
        pl.sql("SELECT * FROM df1 JOIN df2 USING ()").collect()


def test_implicit_join_three_derived_tables() -> None:
    """Implicit join between three aliased derived tables (subqueries)."""
    src_a = pl.DataFrame({"a_id": [1, 2, 3], "x": [10, 20, 30]})
    src_b = pl.DataFrame({"b_id": [1, 2, 3], "y": [100, 200, 300]})
    src_c = pl.DataFrame({"c_id": [1, 2, 3], "z": [1000, 2000, 3000]})

    assert_sql_matches(
        frames={"src_a": src_a, "src_b": src_b, "src_c": src_c},
        query="""
            SELECT a.a_id, a.x, b.y, c.z
            FROM (SELECT * FROM src_a WHERE a_id <= 2) AS a,
                 (SELECT * FROM src_b WHERE b_id >= 1) AS b,
                 (SELECT * FROM src_c WHERE c_id >= 2) AS c
            WHERE a.a_id = b.b_id AND b.b_id = c.c_id
            ORDER BY a.a_id
        """,
        compare_with="sqlite",
        expected={"a_id": [2], "x": [20], "y": [200], "z": [2000]},
    )


def test_implicit_join_mixed_with_left_join() -> None:
    """Implicit join (comma-separated) mixed with an explicit LEFT JOIN."""
    t1 = pl.DataFrame({"a": [1, 2, 3], "v1": [10, 20, 30]})
    t2 = pl.DataFrame({"a": [1, 2, 3], "v2": [100, 200, 300]})
    t3 = pl.DataFrame({"a": [2, 3, 4], "v3": [2000, 3000, 4000]})

    assert_sql_matches(
        frames={"t1": t1, "t2": t2, "t3": t3},
        query="""
            SELECT t1.a, t1.v1, t2.v2, t3.v3
            FROM t1 LEFT JOIN t2 ON t1.a = t2.a, t3
            WHERE t1.a = t3.a
            ORDER BY t1.a
        """,
        compare_with="sqlite",
        expected={
            "a": [2, 3],
            "v1": [20, 30],
            "v2": [200, 300],
            "v3": [2000, 3000],
        },
    )


def test_join_on_between() -> None:
    """BETWEEN / NOT BETWEEN in JOIN ON are handled as join_where predicates."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [15, 25, 35]})
    df2 = pl.DataFrame({"low": [10, 20, 30], "high": [20, 30, 40]})

    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT df1.id, df1.val, df2.low, df2.high
            FROM df1 INNER JOIN df2 ON df1.val BETWEEN df2.low AND df2.high
            ORDER BY df1.id, df2.low
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3],
            "val": [15, 25, 35],
            "low": [10, 20, 30],
            "high": [20, 30, 40],
        },
    )

    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT df1.id, df1.val, df2.low, df2.high
            FROM df1 INNER JOIN df2 ON df1.val NOT BETWEEN df2.low AND df2.high
            ORDER BY df1.id, df2.low
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 1, 2, 2, 3, 3],
            "val": [15, 15, 25, 25, 35, 35],
            "low": [20, 30, 10, 30, 10, 20],
            "high": [30, 40, 20, 40, 20, 30],
        },
    )


def test_join_on_is_not_null() -> None:
    """IS NOT NULL in JOIN ON is passed through as a join_where predicate."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, None, 30]})
    df2 = pl.DataFrame({"id": [1, 2, 3], "x": [100, 200, 300]})

    res = pl.sql("""
        SELECT df1.id, df2.x
        FROM df1 INNER JOIN df2 ON df1.id = df2.id AND df1.val IS NOT NULL
        ORDER BY df1.id
    """).collect()

    expected = pl.DataFrame({"id": [1, 3], "x": [100, 300]})
    assert_frame_equal(res, expected)


def test_join_on_in_list() -> None:
    """IN list in JOIN ON is passed through as a join_where predicate."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "cat": ["A", "B", "C"]})
    df2 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

    res = pl.sql("""
        SELECT df1.id, df2.val
        FROM df1 INNER JOIN df2 ON df1.id = df2.id AND df1.cat IN ('A', 'C')
        ORDER BY df1.id
    """).collect()

    expected = pl.DataFrame({"id": [1, 3], "val": [10, 30]})
    assert_frame_equal(res, expected)


def test_unsupported_operator_in_join_on() -> None:
    """Non-comparison binary operators (e.g. +) in JOIN ON should error."""
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [1, 2, 3]})

    with pytest.raises(SQLInterfaceError, match="unsupported join constraint operator"):
        pl.sql("SELECT * FROM df1 INNER JOIN df2 ON df1.a + df2.b").collect()


@pytest.mark.parametrize("condition", ["ON TRUE", "ON 1 = 1"])
def test_join_on_literal_condition(condition: str) -> None:
    """Tautological literal conditions (ON TRUE, ON 1 = 1) implies CROSS JOIN."""
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [10, 20]})
    res = pl.sql(
        query=f"SELECT * FROM df1 INNER JOIN df2 {condition} ORDER BY a, b",
        eager=True,
    )
    expected = pl.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 10, 20]})
    assert_frame_equal(res, expected)


def test_using_column_not_in_table() -> None:
    """JOIN USING with a non-existent column should produce a clear error."""
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [1, 2]})

    with pytest.raises((SQLInterfaceError, pl.exceptions.ColumnNotFoundError)):
        pl.sql("SELECT * FROM df1 JOIN df2 USING (nonexistent_col)").collect()


def test_join_on_is_null() -> None:
    """IS NULL in JOIN ON is passed through as a join_where predicate."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, None, 30]})
    df2 = pl.DataFrame({"id": [1, 2, 3], "x": [100, 200, 300]})

    res = pl.sql("""
        SELECT df1.id, df2.x
        FROM df1 INNER JOIN df2 ON df1.id = df2.id AND df1.val IS NULL
        ORDER BY df1.id
    """).collect()

    expected = pl.DataFrame({"id": [2], "x": [200]})
    assert_frame_equal(res, expected)


def test_non_equi_between_non_inner_error() -> None:
    """Non-equi BETWEEN condition with LEFT JOIN should error."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [15, 25, 35]})
    df2 = pl.DataFrame({"low": [10, 20, 30], "high": [20, 30, 40]})

    with pytest.raises(SQLInterfaceError, match="non-equi join conditions"):
        pl.sql("""
            SELECT df1.id, df1.val, df2.low, df2.high
            FROM df1 LEFT JOIN df2 ON df1.val BETWEEN df2.low AND df2.high
        """).collect()


def test_using_with_null_values() -> None:
    """USING join where the key column contains NULLs — NULLs should not match."""
    df1 = pl.DataFrame({"key": [1, 2, None], "a": [10, 20, 30]})
    df2 = pl.DataFrame({"key": [1, None, 3], "b": [100, 200, 300]})

    res = pl.sql("""
        SELECT df1.key, df1.a, df2.b FROM df1 INNER JOIN df2 USING (key)
        ORDER BY df1.key
    """).collect()

    expected = pl.DataFrame({"key": [1], "a": [10], "b": [100]})
    assert_frame_equal(res, expected)


def test_multi_table_ambiguous_column_hints() -> None:
    df1 = pl.DataFrame({"id": [1], "a": [10]})
    df2 = pl.DataFrame({"id": [1], "x": [20]})
    df3 = pl.DataFrame({"id": [1], "x": [30]})

    # 'x' exists in df2 and df3 but not df1; the error should mention df2 and df3
    with pytest.raises(SQLInterfaceError, match=r"df2\.x.*df3\.x|df3\.x.*df2\.x"):
        pl.sql("""
            SELECT df1.id, x
            FROM df1
            JOIN df2 ON df1.id = df2.id
            JOIN df3 ON df1.id = df3.id
        """).collect()


def test_ambiguous_column_literal_join_condition() -> None:
    df1 = pl.DataFrame({"id": [1, 2], "val": [10, 20]})
    df2 = pl.DataFrame({"id": [1, 2], "val": [30, 40]})

    with pytest.raises(SQLInterfaceError, match="ambiguous"):
        pl.sql("""
            SELECT * FROM df1 JOIN df2 ON val = 42
        """).collect()


def test_self_join_without_aliases_error() -> None:
    df1 = pl.DataFrame({"id": [1, 2], "val": [10, 20]})

    with pytest.raises(SQLInterfaceError, match="self-join requires table aliases"):
        pl.sql("""
            SELECT * FROM df1 JOIN df1 ON df1.id = df1.val
        """).collect()


def test_implicit_join_self_with_non_equi() -> None:
    """Self-join via implicit syntax with non-equi conditions."""
    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

    res = pl.sql("""
        SELECT a.id, a.val AS a_val, b.val AS b_val
        FROM df AS a, df AS b
        WHERE a.id = b.id AND a.val > 5
        ORDER BY a.id
    """).collect()

    expected = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "a_val": [10, 20, 30],
            "b_val": [10, 20, 30],
        }
    )
    assert_frame_equal(res, expected)


def test_implicit_join_with_residual_single_table_filters() -> None:
    """Non-equi implicit join with residual single-table filters in WHERE."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [1, 2, 3], "score": [100, 200, 300]})

    res = pl.sql("""
        SELECT df1.id, df1.val, df2.score
        FROM df1, df2
        WHERE df1.id = df2.id AND df1.val > 10 AND df2.score < 300
        ORDER BY df1.id
    """).collect()

    expected = pl.DataFrame({"id": [2], "val": [20], "score": [200]})
    assert_frame_equal(res, expected)


def test_join_on_between_column_resolution() -> None:
    """BETWEEN in JOIN ON correctly resolves columns whether they conflict or not."""
    # Tested column unique to the left table
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [15, 25, 35]})
    df2 = pl.DataFrame({"low": [10, 20, 30], "high": [20, 30, 40]})

    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT df1.id, df1.val, df2.low, df2.high
            FROM df1 INNER JOIN df2 ON df1.val BETWEEN df2.low AND df2.high
            ORDER BY df1.id, df2.low
        """,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3],
            "val": [15, 25, 35],
            "low": [10, 20, 30],
            "high": [20, 30, 40],
        },
    )

    # Tested column exists in both tables (must resolve to left, not right)
    df1 = pl.DataFrame({"x": [15, 25, 35]})
    df2 = pl.DataFrame({"x": [99, 99, 99], "low": [10, 20, 30], "high": [20, 30, 40]})

    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT df1.x, df2.low, df2.high
            FROM df1 INNER JOIN df2 ON df1.x BETWEEN df2.low AND df2.high
            ORDER BY df1.x, df2.low
        """,
        compare_with="sqlite",
        expected={
            "x": [15, 25, 35],
            "low": [10, 20, 30],
            "high": [20, 30, 40],
        },
    )


def test_natural_join_no_common_columns_error() -> None:
    """NATURAL JOIN with no common columns should produce a clear error."""
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})

    with pytest.raises(
        SQLInterfaceError, match="no common columns found for NATURAL JOIN"
    ):
        pl.sql("SELECT * FROM df1 NATURAL JOIN df2").collect()


def test_implicit_join_three_tables_conflicting_columns() -> None:
    """Chained 3-table implicit join where all tables share 'id' and 'val'."""
    t1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    t2 = pl.DataFrame({"id": [2, 3, 4], "val": [200, 300, 400]})
    t3 = pl.DataFrame({"id": [3, 4, 5], "val": [3000, 4000, 5000]})

    res = pl.sql("""
        SELECT t1.id, t1.val AS v1, t2.val AS v2, t3.val AS v3
        FROM t1, t2, t3
        WHERE t1.id = t2.id AND t2.id = t3.id
        ORDER BY t1.id
    """).collect()

    expected = pl.DataFrame(
        {
            "id": [3],
            "v1": [30],
            "v2": [300],
            "v3": [3000],
        }
    )
    assert_frame_equal(res, expected)
