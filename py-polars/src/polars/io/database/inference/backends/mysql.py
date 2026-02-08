# ---------------------------------------------------------------------------------
# MySQL database type codes:
# ---------------------------------------------------------------------------------
# In general we can infer the Polars DataType from the type name; certain entries
# that are not reasonably inferable (or are potentially ambiguous) are hardcoded
# with a more self-evident name, with the original commented-out to the right.
# ---------------------------------------------------------------------------------
from enum import IntEnum

# https://github.com/mysql/mysql-server/blob/trunk/include/field_types.h
type_codes = {
    0: "decimal",
    1: "tinyint",  # <- tiny
    2: "smallint",  # <- short
    3: "int4",  # <- long
    4: "float4",  # <- float
    5: "float8",  # <- double
    6: "null",
    7: "timestamp",
    8: "int8",  # <- longlong
    9: "int24",
    10: "date",
    11: "time",
    12: "datetime",
    13: "smallint",  # <- year
    14: "date",  # <- newdate
    15: "varchar",
    16: "bit",
    242: "vector",
    245: "json",
    246: "decimal",  # <- newdecimal
    247: "enum",
    248: "set",
    249: "tiny_blob",
    250: "medium_blob",
    251: "long_blob",
    252: "blob",
    253: "var_string",
    254: "string",
    255: "geometry",
}


# https://github.com/mysql/mysql-server/blob/trunk/include/mysql_com.h
class field_flags(IntEnum):
    """MySQL field flags, as found in the cursor description."""

    NOT_NULL = 1
    PRI_KEY = 2
    UNIQUE_KEY = 4
    MULTIPLE_KEY = 8
    BLOB = 16
    UNSIGNED = 32
    ZEROFILL = 64
    BINARY = 128
    ENUM = 256
    AUTO_INCREMENT = 512
    TIMESTAMP = 1024
    SET = 2048
    NUM = 32768
    PART_KEY = 16384
    GROUP = 32768
    UNIQUE = 65536
