# ---------------------------------------------------------------------------------
# Oracle database type codes:
# ---------------------------------------------------------------------------------
# In general we can infer the Polars DataType from the type name; certain entries
# that are not reasonably inferable (or are potentially ambiguous) are hardcoded
# with a more self-evident name, with the original commented-out to the right.
# ---------------------------------------------------------------------------------


# https://node-oracledb.readthedocs.io/en/latest/api_manual/oracledb.html#oracle-database-type-objects
type_codes = {
    2020: "bfile",
    2008: "float8",  # <- binary_double
    2007: "float4",  # <- binary_float
    2009: "int4",  # <- binary_integer
    2019: "blob",
    2022: "boolean",
    2003: "char",
    2017: "clob",
    2021: "cursor",
    2011: "date",
    2015: "duration",  # interval_ds
    2016: "interval_ym",
    2027: "json",
    2024: "string",  # <- long
    2031: "string",  # <- long_nvarchar
    2025: "binary",  # <- long_raw
    2004: "nchar",
    2018: "nclob",
    2010: "number",
    2002: "nvarchar",
    # 2023: "object",
    2006: "binary",  # <- raw
    2005: "string",  # <- rowid
    2012: "timestamp",
    2014: "timestamp_ltz",
    2013: "timestamp_tz",
    2030: "string",  # <- urowid
    2001: "varchar",
    2033: "vector",
    2032: "xmltype",
}
