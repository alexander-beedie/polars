use sqlparser::derive_dialect;
use sqlparser::dialect::GenericDialect;

derive_dialect!(
    PolarsSQLDialect,
    GenericDialect,
    preserve_type_id = true,
    overrides = { supports_order_by_all = true }
);
