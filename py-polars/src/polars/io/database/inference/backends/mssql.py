# ---------------------------------------------------------------------------------
# MSSQL database type codes:
# ---------------------------------------------------------------------------------
# In general we can infer the Polars DataType from the type name; certain entries
# that are not reasonably inferable (or are potentially ambiguous) are hardcoded
# with a more self-evident name, with the original commented-out to the right.
# ---------------------------------------------------------------------------------

# driver-specific override: unfortunately `pymssql` provides such vague and
# unreliable type codes that this override *cannot* be sensibly enabled.
#
# we strongly recommend use of arrow-odbc or the microsoft-supported
# official integration with pyodbc instead.
# # https://github.com/pymssql/pymssql/blob/master/src/pymssql/_mssql.pyx
#
# pymssql = {
#     1: "string",
#     2: "binary",  # <- overloaded with date fields too
#     3: "number",  # <- too vague (no prec/scale in cursor description)
#     4: "datetime",
#     5: "decimal",  # <- rowid
# }

# https://github.com/FreeTDS/freetds/blob/master/include/freetds/proto.h
type_codes = {
    34: "image",
    35: "text",
    36: "uuid",
    37: "varbinary",
    38: "intn",
    39: "varchar",
    40: "date",
    41: "time",
    42: "datetime2",
    43: "datetimeoffset",
    45: "binary",
    47: "char",
    48: "int1",
    50: "bit",
    52: "int2",
    56: "int4",
    58: "datetim4",
    59: "real",
    60: "money",
    61: "datetime",
    62: "flt8",
    98: "variant",
    103: "nvarchar",
    104: "bitn",
    106: "decimal",
    108: "numeric",
    109: "fltn",
    110: "moneyn",
    111: "datetimn",
    122: "money4",
    127: "int8",
    165: "varbinary",
    167: "varchar",
    173: "binary",
    175: "char",
    231: "nvarchar",
    239: "nchar",
    240: "udt",
    241: "xml",
    243: "table",
}
