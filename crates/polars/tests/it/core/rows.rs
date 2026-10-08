use polars::frame::row::Row;
use polars::prelude::*;

#[test]
fn test_rows_value_not_fitting_column_errors_unless_non_strict() -> PolarsResult<()> {
    let rows = [
        Row::new(vec![AnyValue::Int64(1)]),
        Row::new(vec![AnyValue::String("x")]),
    ];
    assert!(DataFrame::from_rows(&rows).is_err());

    let schema = Schema::from_iter([Field::new("a".into(), DataType::Int64)]);
    assert!(DataFrame::from_rows_and_schema(&rows, &schema, true).is_err());
    let df = DataFrame::from_rows_and_schema(&rows, &schema, false)?;
    assert!(df.equals_missing(&df!["a" => [Some(1i64), None]]?));
    Ok(())
}
