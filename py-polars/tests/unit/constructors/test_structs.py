import pytest

import polars as pl
from polars.exceptions import ComputeError


def test_constructor_non_strict_schema_17956() -> None:
    schema = {
        "logged_event": pl.Struct(
            [
                pl.Field(
                    "completetask",
                    pl.Struct(
                        [
                            pl.Field(
                                "parameters",
                                pl.List(
                                    pl.Struct(
                                        [
                                            pl.Field(
                                                "numericarray",
                                                pl.Struct(
                                                    [
                                                        pl.Field(
                                                            "value", pl.List(pl.Float64)
                                                        ),
                                                    ]
                                                ),
                                            ),
                                        ]
                                    )
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        ),
    }

    data = {
        "logged_event": {
            "completetask": {
                "parameters": [
                    {
                        "numericarray": {
                            "value": [
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                430.5,
                                431,
                                431,
                                431,
                            ]
                        }
                    }
                ]
            }
        }
    }

    lazyframe = pl.LazyFrame(
        [data],
        schema=schema,
        strict=False,
    )
    assert lazyframe.collect().to_dict(as_series=False) == {
        "logged_event": [
            {
                "completetask": {
                    "parameters": [
                        {
                            "numericarray": {
                                "value": [
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    430.5,
                                    431.0,
                                    431.0,
                                    431.0,
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }


def test_series_init_struct_strict() -> None:
    dtype = pl.Struct({"a": pl.Int64})
    with pytest.raises(ComputeError, match="could not append value"):
        pl.Series([{"a": 1.5}], dtype=dtype)

    s = pl.Series([{"a": 1.5}, {"a": "x"}, None], dtype=dtype, strict=False)
    assert s.to_list() == [{"a": 1}, {"a": None}, None]
