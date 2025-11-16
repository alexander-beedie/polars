use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::SQLContext;

#[test]
fn test_literal_only_select_empty_df() {
    // Test case: SELECT with only literals on an empty DataFrame
    let df = DataFrame::empty_with_schema(&Schema::from_iter([
        Field::new("x".into(), DataType::Int64),
        Field::new("y".into(), DataType::Float64),
    ]));

    let mut ctx = SQLContext::new();
    ctx.register("df", df.lazy());

    let result = ctx
        .execute(r#"SELECT 1 AS "one", 2 AS "two" FROM df"#)
        .unwrap()
        .collect()
        .unwrap();

    // Expected: 0 rows (empty DataFrame with correct schema)
    assert_eq!(result.shape(), (0, 2));
    assert_eq!(result.get_column_names(), vec!["one", "two"]);
}

#[test]
fn test_literal_only_select_with_rows() {
    // Test case: SELECT with only literals on a DataFrame with rows
    let df = df! {
        "x" => [1, 2, 3],
        "y" => [4.0, 5.0, 6.0],
    }
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.lazy());

    let result = ctx
        .execute(r#"SELECT 1 AS "one", 2 AS "two" FROM df"#)
        .unwrap()
        .collect()
        .unwrap();

    // Expected: 3 rows with values [1, 2] repeated
    assert_eq!(result.shape(), (3, 2));

    let one_col = result.column("one").unwrap();
    let two_col = result.column("two").unwrap();

    assert_eq!(
        one_col.i32().unwrap().to_vec(),
        vec![Some(1), Some(1), Some(1)]
    );
    assert_eq!(
        two_col.i32().unwrap().to_vec(),
        vec![Some(2), Some(2), Some(2)]
    );
}

#[test]
fn test_literal_with_column_reference() {
    // Test case: SELECT with both literals and column references (should not be affected)
    let df = df! {
        "x" => [1i64, 2i64, 3i64],
    }
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.lazy());

    let result = ctx
        .execute(r#"SELECT x, 42 AS "literal" FROM df"#)
        .unwrap()
        .collect()
        .unwrap();

    // Expected: 3 rows with x values and literal 42
    assert_eq!(result.shape(), (3, 2));

    let x_col = result.column("x").unwrap();
    let lit_col = result.column("literal").unwrap();

    assert_eq!(
        x_col.i64().unwrap().to_vec(),
        vec![Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        lit_col.i32().unwrap().to_vec(),
        vec![Some(42), Some(42), Some(42)]
    );
}

#[test]
fn test_literal_only_with_expressions() {
    // Test case: SELECT with literal expressions (no column references)
    let df = df! {
        "x" => [1i64, 2i64],
    }
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.lazy());

    let result = ctx
        .execute(r#"SELECT 1 + 2 AS "sum", 10 * 5 AS "product" FROM df"#)
        .unwrap()
        .collect()
        .unwrap();

    // Expected: 2 rows with computed literal values
    assert_eq!(result.shape(), (2, 2));

    let sum_col = result.column("sum").unwrap();
    let prod_col = result.column("product").unwrap();

    assert_eq!(sum_col.i32().unwrap().to_vec(), vec![Some(3), Some(3)]);
    assert_eq!(prod_col.i32().unwrap().to_vec(), vec![Some(50), Some(50)]);
}
