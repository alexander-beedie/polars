use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_sample_df() -> LazyFrame {
    df! {
        "id" => [1, 2, 3, 4, 5, 6],
        "category" => ["A", "A", "A", "B", "B", "B"],
        "value" => [100, 200, 150, 300, 250, 400]
    }
    .unwrap()
    .lazy()
}

#[test]
fn test_qualify_with_avg_window() {
    // Test QUALIFY with AVG window function
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value > AVG(value) OVER (PARTITION BY category)
        ORDER BY category, value
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Category A avg: 150, above avg: 200 (1 row)
    // Category B avg: 316.67, above avg: 400 (1 row)
    assert_eq!(result.height(), 2);
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(0),
        Some(200)
    );
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(1),
        Some(400)
    );
}

#[test]
fn test_qualify_with_sum_window() {
    // Test QUALIFY with SUM window function
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY SUM(value) OVER (PARTITION BY category) > 400
        ORDER BY id
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Category A sum: 450 (>400) -> 3 rows
    // Category B sum: 950 (>400) -> 3 rows
    assert_eq!(result.height(), 6);
}

#[test]
fn test_qualify_with_max_window() {
    // Test QUALIFY with MAX window function
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value = MAX(value) OVER (PARTITION BY category)
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Category A max: 200, Category B max: 400
    assert_eq!(result.height(), 2);
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(0),
        Some(200)
    );
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(1),
        Some(400)
    );
}

#[test]
fn test_qualify_with_min_window() {
    // Test QUALIFY with MIN window function
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value = MIN(value) OVER (PARTITION BY category)
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Category A min: 100, Category B min: 250
    assert_eq!(result.height(), 2);
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(0),
        Some(100)
    );
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(1),
        Some(250)
    );
}

#[test]
fn test_qualify_with_count_window() {
    // Test QUALIFY with COUNT window function
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY COUNT(*) OVER (PARTITION BY category) = 3
        ORDER BY id
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Both categories have 3 rows each
    assert_eq!(result.height(), 6);
}

#[test]
fn test_qualify_with_complex_expression() {
    // Test QUALIFY with complex boolean expression
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY
            value > AVG(value) OVER (PARTITION BY category)
            AND value < 500
        ORDER BY category, value
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Category A: value=200 (>150 avg, <500) ✓
    // Category B: value=400 (>316.67 avg, <500) ✓
    assert_eq!(result.height(), 2);
}

#[test]
fn test_qualify_with_where_clause() {
    // Test QUALIFY combined with WHERE
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        WHERE value > 100
        QUALIFY value = MAX(value) OVER (PARTITION BY category)
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // WHERE filters first, then window functions, then QUALIFY
    // After WHERE: id=2,3,4,5,6
    // Max per category: A=200, B=400
    assert_eq!(result.height(), 2);
}

// Note: This test is temporarily disabled due to streaming engine feature requirements
// #[test]
// fn test_qualify_with_group_by_and_having() {
//     // Test QUALIFY with GROUP BY and HAVING
//     let df = df! {
//         "category" => ["A", "A", "B", "B", "C", "C"],
//         "subcategory" => ["X", "Y", "X", "Y", "X", "Y"],
//         "amount" => [100, 200, 150, 250, 300, 100]
//     }
//     .unwrap()
//     .lazy();

//     let mut ctx = SQLContext::new();
//     ctx.register("df", df);

//     let sql = r#"
//         SELECT
//             category,
//             SUM(amount) as total
//         FROM df
//         GROUP BY category
//         HAVING total > 200
//         QUALIFY total = MAX(total) OVER (PARTITION BY category)
//         ORDER BY category
//     "#;

//     let result = ctx.execute(sql).unwrap().collect().unwrap();

//     // After GROUP BY: A=300, B=400, C=400
//     // After HAVING (>200): A=300, B=400, C=400
//     // QUALIFY: total = MAX(total) within partition -> all rows pass
//     assert_eq!(result.height(), 3);
// }

#[test]
fn test_qualify_without_partition() {
    // Test QUALIFY with simple expression and partitioned window
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value >= 300
        AND SUM(value) OVER (PARTITION BY category) > 500
        ORDER BY value
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Category B sum: 950 (>500), values >= 300: 300, 400
    assert_eq!(result.height(), 2);
}

#[test]
fn test_qualify_with_multiple_window_functions() {
    // Test QUALIFY with multiple window functions in expression
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY
            value = MAX(value) OVER (PARTITION BY category)
            OR value = MIN(value) OVER (PARTITION BY category)
        ORDER BY id
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Max per category: 200 (A), 400 (B)
    // Min per category: 100 (A), 250 (B)
    // Result: id=1, 2, 5, 6
    assert_eq!(result.height(), 4);
}

#[test]
fn test_qualify_returns_empty_when_no_match() {
    // Test that QUALIFY returns empty result when no rows match
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value > MAX(value) OVER (PARTITION BY category)
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();
    // No value can be greater than the max in its partition
    assert_eq!(result.height(), 0);
}

#[test]
fn test_qualify_with_alias_from_select() {
    // Test QUALIFY referencing an aliased window function from SELECT
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT
            id,
            category,
            value,
            MAX(value) OVER (PARTITION BY category) as max_value
        FROM df
        QUALIFY value = max_value
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Should return rows where value equals the max in their category
    // Category A: value=200, Category B: value=400
    assert_eq!(result.height(), 2);
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(0),
        Some(200)
    );
    assert_eq!(
        result.column("value").unwrap().i32().unwrap().get(1),
        Some(400)
    );

    // Check that max_value column exists in output
    assert!(result.column("max_value").is_ok());
}

#[test]
fn test_qualify_mixed_alias_and_explicit() {
    // Test QUALIFY with both alias reference and explicit window function
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT
            id,
            category,
            value,
            AVG(value) OVER (PARTITION BY category) as avg_value
        FROM df
        QUALIFY value > avg_value AND COUNT(*) OVER (PARTITION BY category) = 3
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Values above category average, in categories with 3 rows
    assert_eq!(result.height(), 2);
}

#[test]
fn test_qualify_without_window_function_errors() {
    // QUALIFY without window functions should fail
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value > 200
    "#;

    let result = ctx.execute(sql);
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("window function") || err_msg.contains("QUALIFY"),
        "Expected error about window functions in QUALIFY, got: {}",
        err_msg
    );
}

#[test]
fn test_qualify_with_distinct() {
    // Test QUALIFY combined with DISTINCT
    let df = df! {
        "id" => [1, 2, 3, 4, 5, 6],
        "category" => ["A", "A", "B", "B", "C", "C"],
        "value" => [100, 100, 200, 200, 300, 300]
    }
    .unwrap()
    .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT DISTINCT category, value
        FROM df
        QUALIFY value = MAX(value) OVER (PARTITION BY category)
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Max per category: A=100, B=200, C=300
    // After DISTINCT: one row per category
    assert_eq!(result.height(), 3);
}

#[test]
fn test_qualify_with_cumulative_sum() {
    // Test QUALIFY with cumulative window function
    let df = df! {
        "id" => [1, 2, 3, 4, 5],
        "value" => [10, 20, 30, 40, 50]
    }
    .unwrap()
    .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, value
        FROM df
        QUALIFY SUM(value) OVER (ORDER BY id) <= 60
        ORDER BY id
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Cumulative: 10, 30, 60, 100, 150
    // <= 60: first 3 rows
    assert_eq!(result.height(), 3);
}

#[test]
fn test_qualify_with_window_in_select() {
    // Test QUALIFY when window function also appears in SELECT
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT
            id,
            category,
            value,
            AVG(value) OVER (PARTITION BY category) as avg_value
        FROM df
        QUALIFY value > AVG(value) OVER (PARTITION BY category)
        ORDER BY category
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // Values above their category average
    assert_eq!(result.height(), 2);
    // Check that avg_value column exists
    assert!(result.column("avg_value").is_ok());
}

#[test]
fn test_qualify_all_rows_filtered() {
    // Test when QUALIFY filters out all rows
    let df = create_sample_df();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let sql = r#"
        SELECT id, category, value
        FROM df
        QUALIFY value < MIN(value) OVER (PARTITION BY category)
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // No value can be less than the minimum in its partition
    assert_eq!(result.height(), 0);
}

#[test]
fn test_qualify_with_alias_and_comparison() {
    // Test QUALIFY referencing a window function alias with comparison
    let df = df! {
        "c1" => [1, 2, 3, 4, 5, 6],
        "c2" => ["A", "A", "B", "B", "C", "C"],
        "c3" => [10, 20, 30, 40, 50, 60]
    }
    .unwrap()
    .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("t1", df);

    let sql = r#"
        SELECT c2, c3, SUM(c3) OVER (PARTITION BY c2) as total
        FROM t1
        QUALIFY total > 50
        ORDER BY c2
    "#;

    let result = ctx.execute(sql).unwrap().collect().unwrap();

    // c2="A": SUM=30 (not > 50)
    // c2="B": SUM=70 (> 50) ✓
    // c2="C": SUM=110 (> 50) ✓
    assert_eq!(result.height(), 4);
}
