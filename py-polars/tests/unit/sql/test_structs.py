from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import StructFieldNotFoundError
from polars.testing import assert_frame_equal


@pytest.fixture()
def struct_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [100, 200, 300, 400],
            "name": ["Alice", "Bob", "David", "Zoe"],
            "age": [32, 27, 19, 45],
            "other": [{"n": 1.5}, {"n": None}, {"n": -0.5}, {"n": 2.0}],
        }
    ).select(pl.struct(pl.all()).alias("json_msg"))


def test_struct_field_selection(struct_df: pl.DataFrame) -> None:
    res = struct_df.sql(
        """
        SELECT
          -- validate table alias resolution
          frame.json_msg.id AS ID,
          self.json_msg.name AS NAME,
          json_msg.age AS AGE
        FROM
          self AS frame
        WHERE
          json_msg.age > 20 AND
          json_msg.other.n IS NOT NULL  -- note: nested struct field
        ORDER BY
          json_msg.name DESC
        """
    )

    expected = pl.DataFrame(
        {
            "ID": [400, 100],
            "NAME": ["Zoe", "Alice"],
            "AGE": [45, 32],
        }
    )
    assert_frame_equal(expected, res)


@pytest.mark.parametrize(
    "invalid_column",
    [
        "json_msg.invalid_column",
        "json_msg.other.invalid_column",
        "self.json_msg.other.invalid_column",
    ],
)
def test_struct_indexing_errors(invalid_column: str, struct_df: pl.DataFrame) -> None:
    with pytest.raises(StructFieldNotFoundError, match="invalid_column"):
        struct_df.sql(f"SELECT {invalid_column} FROM self")


@pytest.mark.parametrize(
    "join_constraint",
    [
        "ON df1.json_msg.id = df2.json_msg.id",
        "USING (json_msg.id)",
    ],
)
def test_struct_join_constraints(join_constraint: str, struct_df: pl.DataFrame) -> None:
    df1, df2 = (  # noqa: F841
        struct_df,
        pl.DataFrame(
            {
                "id": [600, 200, 500, 400],
                "age": [42, 27, 20, 45],
            }
        ).select(pl.struct(pl.all()).alias("json_msg")),
    )

    res = pl.sql(f"""
        SELECT df1.json_msg.* EXCLUDE other
        FROM df1 LEFT SEMI JOIN df2 {join_constraint}
    """)
    expected = pl.DataFrame(
        {
            "id": [200, 400],
            "name": ["Bob", "Zoe"],
            "age": [27, 45],
        }
    ).select(pl.struct(pl.all()).alias("json_msg"))
    assert_frame_equal(expected, res)
