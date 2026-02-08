from __future__ import annotations

import functools
import re
from contextlib import suppress
from importlib import import_module
from inspect import isclass
from math import ceil
from typing import TYPE_CHECKING, Any

from polars.datatypes import (
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    Object,
    String,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from polars.datatypes._parse import parse_py_type_into_dtype
from polars.datatypes.group import (
    INTEGER_DTYPES,
    SIGNED_INTEGER_DTYPES,
    SIGNED_UNSIGNED_INTEGER_LOOKUP,
    UNSIGNED_INTEGER_DTYPES,
)

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

DRIVER_TYPES_LOOKUP: dict[tuple[str, str], dict[Any, str]] = {}


# @functools.lru_cache(64)
def infer_dtype_from_db_typename(
    value: str,
    *,
    raise_unmatched: bool = False,
) -> PolarsDataType | None:
    """
    Attempt to infer Polars dtype from database cursor `type_code` string value.

    Parameters
    ----------
    value : str
        A database-specific type name.
    raise_unmatched : bool, optional
        If True, raise a ValueError if the type name cannot be matched.

    Examples
    --------
    >>> infer_dtype_from_db_typename("INT2")
    Int16
    >>> infer_dtype_from_db_typename("NVARCHAR")
    String
    >>> infer_dtype_from_db_typename("NUMERIC(10,2)")
    Decimal(precision=10, scale=2)
    >>> infer_dtype_from_db_typename("TIMESTAMP WITHOUT TZ")
    Datetime(time_unit='us', time_zone=None)
    """
    dtype: PolarsDataType | None = None

    # normalise string name/case (eg: 'IntegerType' -> 'INTEGER')
    original_value = value
    value = value.upper().replace("TYPE", "").replace("DB_", "").strip("_")

    # early exit on 'range' (not returnable; creates ranges of other types)
    if value.endswith(("RANGE", "RANGE[]")):
        return None

    # extract optional type modifier (eg: 'VARCHAR(64)' -> '64')
    if re.search(r"\([\w,: ]+\)$", value):
        modifier = value[value.find("(") + 1 : -1]
        value = value.split("(")[0]
    elif (
        not value.startswith(("<", ">")) and re.search(r"\[[\w,\]\[: ]+]$", value)
    ) or value.endswith(("[S]", "[MS]", "[US]", "[NS]")):
        modifier = value[value.find("[") + 1 : -1]
        value = value.split("[")[0]
    else:
        modifier = ""

    # array dtypes
    array_aliases = ("ARRAY", "LIST", "VECTOR", "[]")
    if value.endswith(array_aliases) or value.startswith(array_aliases):
        for a in array_aliases:
            value = value.replace(a, "", 1) if value else ""

        nested: PolarsDataType | None = None
        if not value and modifier:
            nested = infer_dtype_from_db_typename(value=modifier)
        else:
            if inner_value := infer_dtype_from_db_typename(
                value[1:-1]
                if (value[0], value[-1]) == ("<", ">")
                else re.sub(r"\W", "", re.sub(r"\WOF\W", "", value))
            ):
                nested = inner_value
            elif modifier:
                nested = infer_dtype_from_db_typename(value=modifier)
        if nested:
            dtype = List(nested)

    # float dtypes
    elif value.startswith("FLOAT") or ("DOUBLE" in value) or (value == "REAL"):
        dtype = (
            Float32
            if value == "FLOAT4"
            or (value.endswith(("16", "32")) or (modifier in ("16", "32")))
            else Float64
        )

    # integer dtypes
    elif ("INTERVAL" not in value and "RANGE" not in value) and (
        value.startswith(("INT", "UINT", "UNSIGNED"))
        or value.endswith(("INT", "SERIAL"))
        or ("INTEGER" in value)
        or value in ("ROWID", "LONG", "LONGLONG")
    ):
        sz: int | str | None = None
        if unsigned := (
            value.startswith("UINT")
            or ("U" in value and "MEDIUM" not in value)
            or ("UNSIGNED" in value)
            or value == "ROWID"
        ):
            value = value.replace("UINT", "INT")

        if "LARGE" in value or value.startswith("BIG") or value in ("INT8", "LONGLONG"):
            sz = 64
        elif "MEDIUM" in value or value in ("INT3", "INT4", "INT24", "LONG", "SERIAL"):
            sz = 32
        elif "SMALL" in value or value in ("INT2", "SHORT"):
            sz = 16
        elif "TINY" in value or value == "INT1":
            sz = 8

        sz = modifier if (not sz and modifier) else sz
        if not isinstance(sz, int):
            sz = int(sz) if isinstance(sz, str) and sz.isdigit() else None

        dtype = integer_dtype_from_nbits(
            bits=sz,
            unsigned=unsigned,
            default=UInt64 if unsigned else Int64,
        )

    # number types (note: 'number' alone is not that helpful and requires refinement)
    elif "NUMBER" in value and "CARDINAL" in value:
        dtype = UInt64

    # decimal dtypes
    elif (is_dec := ("DECIMAL" in value)) or ("NUMERIC" in value):
        if "," in modifier:
            prec, scale = modifier.split(",")
            dtype = Decimal(int(prec), int(scale))
        else:
            dtype = Decimal if is_dec else Float64

    # string dtypes
    elif (
        any(tp in value for tp in ("VARCHAR", "STRING", "TEXT", "UNICODE", "XML"))
        or value.startswith(("STR", "CHAR", "BPCHAR", "NCHAR", "UTF"))
        or value.endswith(("_UTF8", "_UTF16", "_UTF32"))
    ):
        dtype = String

    # binary dtypes
    elif (
        re.match("(BYTE[AS]|[BC]LOB)", value)
        or value.endswith("_BLOB")
        or value == "BINARY"
    ):
        dtype = Binary

    # boolean dtypes
    elif value.startswith("BOOL"):
        dtype = Boolean

    # null dtype (unusual, but valid)
    elif value == "NULL":
        dtype = Null

    # object dtype (we set this)
    elif value == "OBJECT":
        dtype = Object

    # temporal dtypes
    elif value.startswith(("DATETIME", "TIMESTAMP")) and not (value.endswith("[D]")):
        if any((tz in value.replace(" ", "")) for tz in ("TZ", "TIMEZONE")):
            if "WITHOUT" not in value:
                return None  # there's a timezone, but we don't know what it is
        unit = timeunit_from_precision(modifier) if modifier else "us"
        dtype = Datetime(time_unit=(unit or "us"))  # type: ignore[arg-type]
    else:
        value = re.sub(r"\d", "", value)
        if value in ("INTERVAL", "TIMEDELTA", "DURATION"):
            dtype = Duration
        elif value == "DATE":
            dtype = Date
        elif value == "TIME":
            dtype = Time

    if not dtype and raise_unmatched:
        msg = f"cannot infer dtype from {original_value!r} string value"
        raise ValueError(msg)

    return dtype


# @functools.lru_cache(64)
def _infer_dtype(
    driver_name: str,
    type_code: Any,
    internal_size: int | None,
    precision: int | None,
    scale: int | None,
    *extras: Any,
) -> PolarsDataType | None:
    """Infer dtype from driver/backend specific introspection."""
    dtype: PolarsDataType | None = None
    backend = _backend_from_driver(driver_name)

    if backend is None:
        return None
    try:
        backend_mod = import_module(f"polars.io.database.inference.backends.{backend}")
        type_codes = getattr(backend_mod, "type_codes", {}).copy()
        type_codes.update(getattr(backend_mod, driver_name, {}))
        if driver_name == "oracledb":
            type_code = type_code.num

        if tp_name := type_codes.get(type_code):
            if (tp := infer_dtype_from_db_typename(tp_name)) is not None:
                dtype = tp
            elif tp_name == "number" and precision is not None and scale is not None:
                if precision > 0 and scale >= 0:
                    dtype = Decimal(precision=precision, scale=scale)

        # special-case; one of the few backends that supports unsigned ints
        if backend == "mysql" and extras and dtype in SIGNED_INTEGER_DTYPES:
            field_flags = backend_mod.field_flags
            if (flags := extras[0]) and flags & getattr(field_flags, "UNSIGNED", 0):
                if dtype in SIGNED_INTEGER_DTYPES:
                    dtype = SIGNED_UNSIGNED_INTEGER_LOOKUP[dtype]

        return dtype  # noqa: TRY300
    except ModuleNotFoundError:
        return None


def _backend_driver_from_alchemy_cursor(cursor: Any) -> tuple[str, str]:
    """Return the underlying driver name from a SQLAlchemy cursor object."""
    cursor_type = type(cursor).__name__
    if cursor_type == "Session":
        engine = cursor.bind.engine
    elif cursor_type == "async_sessionmaker":
        engine = cursor.kw["bind"].engine
    else:
        engine = cursor.engine
    return engine.dialect.name, engine.driver


@functools.lru_cache(16)
def _backend_from_driver(driver_name: str) -> str | None:
    """Return the backend associated with a specific driver."""
    backend: str | None = None
    if driver_name in ("asyncpg", "psycopg", "psycopg2"):
        backend = "postgresql"
    elif driver_name in ("mysqlconnector", "mariadbconnector"):
        backend = "mysql"
    elif driver_name == "pymssql":
        backend = "mssql"
    elif driver_name == "oracledb":
        backend = "oracle"
    return backend


def _refine_dtype(
    backend: str | None,
    driver_name: str,
    dtype: PolarsDataType | None,
    internal_size: int | None,
    precision: int | None,
    scale: int | None,
) -> PolarsDataType | None:
    """Refine dtype based on additional cursor description attributes."""
    if dtype is None:
        return None
    elif dtype == Float64 and internal_size in (4, 24):
        dtype = Float32
    elif dtype in INTEGER_DTYPES:
        if driver_name == "pyodbc":
            dtype = {
                19: Int64,
                10: Int32,
                5: Int16,
                3: (UInt8 if backend == "mssql" else Int8),
            }.get(precision, dtype)  # type: ignore[arg-type]

        elif internal_size in (2, 4, 8):
            bits = internal_size * 8
            dtype = integer_dtype_from_nbits(
                bits,
                unsigned=(dtype in UNSIGNED_INTEGER_DTYPES),
                default=dtype,
            )
    elif (
        dtype == Decimal
        and isinstance(precision, int)
        and isinstance(scale, int)
        and precision <= 38
        and scale <= 38
    ):
        dtype = Decimal(precision, scale)

    return dtype


def infer_dtype_from_cursor_description(
    description: tuple[Any, ...],
    backend: str | None,
    driver_name: str,
) -> PolarsDataType | None:
    """
    Infer Polars dtype from database cursor description `type_code` attribute.

    Parameters
    ----------
    description : tuple
        DBAPI2 cursor description attributes.
    backend : str
        Name of the database backend (eg: 'postgresql', 'mysql', etc).
    driver_name : str
        Name of the database driver module (eg: 'psycopg2', 'redshift_connector', etc).
    """
    dtype: PolarsDataType | None
    type_code, _disp_size, internal_size, precision, scale, _null_ok, *extras = (
        description
    )
    if isclass(type_code):
        # python types, eg: int, float, str, etc
        with suppress(TypeError):
            dtype = _refine_dtype(
                backend,
                driver_name,
                parse_py_type_into_dtype(type_code, raise_unmatched=False),  # type: ignore[arg-type]
                internal_size,
                precision,
                scale,
            )

    elif isinstance(type_code, str):
        # database/sql type names, eg: "VARCHAR", "NUMERIC", "BLOB", etc
        dtype = _refine_dtype(
            backend,
            driver_name,
            infer_dtype_from_db_typename(value=type_code),
            internal_size,
            precision,
            scale,
        )
    else:
        # DBAPI2 says nothing about how to define the type_code; if it wasn't defined
        # as a string or python class it can be defined as almost *anything*. by
        # introspecting the driver module we can infer additional information.
        dtype = _infer_dtype(
            driver_name,
            type_code,
            internal_size,
            precision,
            scale,
            *extras,
        )
    return dtype


@functools.lru_cache(8)
def integer_dtype_from_nbits(
    bits: int,
    *,
    unsigned: bool,
    default: PolarsDataType | None = None,
) -> PolarsDataType | None:
    """
    Return matching Polars integer dtype from num bits and signed/unsigned flag.

    Examples
    --------
    >>> integer_dtype_from_nbits(8, unsigned=False)
    Int8
    >>> integer_dtype_from_nbits(32, unsigned=True)
    UInt32
    """
    dtype = {
        (8, False): Int8,
        (8, True): UInt8,
        (16, False): Int16,
        (16, True): UInt16,
        (32, False): Int32,
        (32, True): UInt32,
        (64, False): Int64,
        (64, True): UInt64,
    }.get((bits, unsigned))

    if dtype is None and default is not None:
        return default
    return dtype


def timeunit_from_precision(precision: int | str | None) -> str | None:
    """
    Return `time_unit` from integer precision value.

    Examples
    --------
    >>> timeunit_from_precision(3)
    'ms'
    >>> timeunit_from_precision(5)
    'us'
    >>> timeunit_from_precision(7)
    'ns'
    """
    if not precision:
        return None
    elif isinstance(precision, str):
        if precision.isdigit():
            precision = int(precision)
        elif (precision := precision.lower()) in ("s", "ms", "us", "ns"):
            return "ms" if precision == "s" else precision
    try:
        n = min(max(3, ceil(precision / 3) * 3), 9)  # type: ignore[operator]
        return {3: "ms", 6: "us", 9: "ns"}.get(n)
    except TypeError:
        return None
