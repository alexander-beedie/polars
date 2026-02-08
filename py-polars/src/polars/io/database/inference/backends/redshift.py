# ---------------------------------------------------------------------------------
# Redshift database type codes:
# ---------------------------------------------------------------------------------
# In general we can infer the Polars DataType from the type name; certain entries
# that are not reasonably inferable (or are potentially ambiguous) are hardcoded
# with a more self-evident name, with the original commented-out to the right.
# ---------------------------------------------------------------------------------

# https://github.com/aws/amazon-redshift-python-driver/blob/master/redshift_connector/utils/oids.py
type_codes = {
    -1: "nulltype",
    16: "boolean",
    17: "bytes",
    18: "char",
    19: "string",  # <- name
    20: "bigint",
    21: "smallint",
    22: "smallint_vector",
    23: "integer",
    24: "string",  # <- regproc
    25: "text",
    26: "uint4",  # <- oid
    28: "uint4",  # <- xid
    114: "json",
    199: "json_array",
    600: "point",
    650: "cidr",
    651: "cidr_array",
    700: "real",
    701: "float",
    702: "abstime",
    705: "unknown",
    790: "money",
    791: "money_array",
    829: "macaddr",
    869: "inet",
    1000: "boolean_array",
    1001: "bytes_array",
    1002: "char_array",
    1003: "name_array",
    1005: "smallint_array",
    1007: "integer_array",
    1009: "text_array",
    1014: "bpchar_array",
    1015: "varchar_array",
    1016: "bigint_array",
    1021: "real_array",
    1022: "float_array",
    1028: "oid_array",
    1033: "string",  # <- aclitem
    1034: "string_array",  # <- aclitem_array
    1041: "inet_array",
    1042: "bpchar",
    1043: "varchar",
    1082: "date",
    1083: "time",
    1114: "timestamp",
    1115: "timestamp_array",
    1182: "date_array",
    1183: "time_array",
    1184: "timestamptz",
    1185: "timestamptz_array",
    1186: "interval",
    1187: "interval_array",
    1188: "intervaly2m",
    1189: "intervaly2m_array",
    1190: "intervald2s",
    1191: "intervald2s_array",
    1231: "numeric_array",
    1263: "cstring_array",
    1266: "timetz",
    1700: "numeric",
    2275: "cstring",
    2277: "any_array",
    2950: "uuid_type",
    2951: "uuid_array",
    3000: "geometry",
    3001: "geography",
    3802: "jsonb",
    3807: "jsonb_array",
    3999: "geometryhex",
    4000: "super",
    6551: "varbyte",
}
