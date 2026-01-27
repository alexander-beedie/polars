from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "value": [100, 200, 150, 250, 120, 220, 180, 280, 130, 230],
        }
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (  # Basic FETCH FIRST / NEXT variants
            "SELECT * FROM self ORDER BY id FETCH FIRST 3 ROWS ONLY",
            {"id": [1, 2, 3], "category": ["A", "B", "A"], "value": [100, 200, 150]},
        ),
        (
            "SELECT * FROM self ORDER BY id FETCH NEXT 3 ROWS ONLY",
            {"id": [1, 2, 3], "category": ["A", "B", "A"], "value": [100, 200, 150]},
        ),
        (
            "SELECT * FROM self ORDER BY id FETCH FIRST 1 ROW ONLY",
            {"id": [1], "category": ["A"], "value": [100]},
        ),
        (  # OFFSET with FETCH
            "SELECT * FROM self ORDER BY id OFFSET 8 ROWS FETCH FIRST 5 ROWS ONLY",
            {"id": [9, 10], "category": ["A", "B"], "value": [130, 230]},
        ),
        (  # FETCH with WHERE
            "SELECT * FROM self WHERE category = 'A' ORDER BY value FETCH FIRST 3 ROWS ONLY",
            {"id": [1, 5, 9], "category": ["A", "A", "A"], "value": [100, 120, 130]},
        ),
        (  # FETCH with GROUP BY
            "SELECT category, SUM(value) AS total FROM self GROUP BY category ORDER BY total DESC FETCH FIRST 1 ROW ONLY",
            {"category": ["B"], "total": [1180]},
        ),
        (  # FETCH with DISTINCT
            "SELECT DISTINCT category FROM self ORDER BY category FETCH FIRST 1 ROW ONLY",
            {"category": ["A"]},
        ),
        (  # FETCH in subquery
            "SELECT * FROM (SELECT * FROM self ORDER BY value DESC FETCH FIRST 3 ROWS ONLY) AS top3 ORDER BY id",
            {"id": [4, 8, 10], "category": ["B", "B", "B"], "value": [250, 280, 230]},
        ),
        (  # FETCH with CTE (top 5 by value are all category B: 280,250,230,220,200)
            "WITH top5 AS (SELECT * FROM self ORDER BY value DESC FETCH FIRST 5 ROWS ONLY) SELECT category, COUNT(*) AS cnt FROM top5 GROUP BY category ORDER BY category",
            {"category": ["B"], "cnt": [5]},
        ),
        (  # Queries that should return no rows
            "SELECT * FROM self FETCH FIRST 0 ROWS ONLY",
            {"id": [], "category": [], "value": []},
        ),
        (
            "SELECT * FROM self ORDER BY id OFFSET 100 ROWS FETCH FIRST 5 ROWS ONLY",
            {"id": [], "category": [], "value": []},
        ),
    ],
)
def test_fetch_clause(
    df: pl.DataFrame, query: str, expected: dict[str, Sequence[Any]]
) -> None:
    """Test FETCH clause in various SQL contexts."""
    assert_sql_matches(
        df,
        query=query,
        compare_with="duckdb",
        expected=expected,
    )


def test_fetch_with_join(df: pl.DataFrame) -> None:
    """Test FETCH with JOIN operations."""
    categories = pl.DataFrame(
        {"category": ["A", "B"], "description": ["Alpha", "Beta"]}
    )
    assert_sql_matches(
        frames={
            "test": df,
            "categories": categories,
        },
        query="""
            SELECT test.id, test.value, categories.description
            FROM test
            JOIN categories ON test.category = categories.category
            ORDER BY test.value DESC
            FETCH FIRST 3 ROWS ONLY
        """,
        compare_with="duckdb",
        expected={
            "id": [8, 4, 10],
            "value": [280, 250, 230],
            "description": ["Beta", "Beta", "Beta"],
        },
    )


def test_fetch_with_union(df: pl.DataFrame) -> None:
    """Test FETCH with UNION operations."""
    assert_sql_matches(
        frames={"tbl": df},
        query="""
            SELECT id, value FROM tbl WHERE category = 'A'
            UNION ALL
            SELECT id, value FROM tbl WHERE category = 'B'
            ORDER BY value
            FETCH FIRST 5 ROWS ONLY
        """,
        expected={"id": [1, 5, 9, 3, 7], "value": [100, 120, 130, 150, 180]},
        compare_with="duckdb",
    )


@pytest.mark.parametrize(
    ("query", "error_type", "match"),
    [
        (
            "SELECT * FROM self FETCH FIRST 50 PERCENT ROWS ONLY",
            SQLInterfaceError,
            r"`FETCH` with `PERCENT` is not supported",
        ),
        (
            "SELECT * FROM self ORDER BY value FETCH FIRST 5 ROWS WITH TIES",
            SQLInterfaceError,
            r"`FETCH` with `WITH TIES` is not supported",
        ),
        (
            "SELECT * FROM self LIMIT 5 FETCH FIRST 3 ROWS ONLY",
            SQLSyntaxError,
            r"cannot use both `LIMIT` and `FETCH`",
        ),
    ],
)
def test_fetch_errors(
    df: pl.DataFrame, query: str, error_type: type[Exception], match: str
) -> None:
    """Test error conditions for unsupported FETCH features."""
    with pytest.raises(error_type, match=match):
        df.sql(query)
