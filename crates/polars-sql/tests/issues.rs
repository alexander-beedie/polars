use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
#[cfg(feature = "csv")]
fn iss_7437() -> PolarsResult<()> {
    let mut context = SQLContext::new();
    let sql = r#"
        CREATE TABLE foods AS
        SELECT *
        FROM read_csv('../../examples/datasets/foods1.csv')"#;
    context.execute(sql)?.collect()?;

    let df_sql = context
        .execute(
            r#"
            SELECT "category" as category
            FROM foods
            GROUP BY "category"
        "#,
        )?
        .collect()?
        .sort(["category"], SortMultipleOptions::default())?;

    let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
        .finish()?
        .group_by(vec![col("category").alias("category")])
        .agg(vec![])
        .collect()?
        .sort(["category"], Default::default())?;

    assert!(df_sql.equals(&expected));
    Ok(())
}

#[test]
#[cfg(feature = "csv")]
fn iss_7436() {
    let mut context = SQLContext::new();
    let sql = r#"
        CREATE TABLE foods AS
        SELECT *
        FROM read_csv('../../examples/datasets/foods1.csv')"#;
    context.execute(sql).unwrap().collect().unwrap();
    let df_sql = context
        .execute(
            r#"
        SELECT
            "fats_g" AS fats,
            AVG(calories) OVER (PARTITION BY "category") AS avg_calories_by_category
        FROM foods
        LIMIT 5
        "#,
        )
        .unwrap()
        .collect()
        .unwrap();
    let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
        .finish()
        .unwrap()
        .select(&[
            col("fats_g").alias("fats"),
            col("calories")
                .mean()
                .over(vec![col("category")])
                .alias("avg_calories_by_category"),
        ])
        .limit(5)
        .collect()
        .unwrap();
    assert!(df_sql.equals(&expected));
}

#[test]
fn iss_7440() {
    let df = df! {
        "a" => [2.0, -2.5]
    }
    .unwrap()
    .lazy();
    let sql = r#"SELECT a, FLOOR(a) AS floor, CEIL(a) AS ceil FROM df"#;
    let mut context = SQLContext::new();
    context.register("df", df.clone());

    let df_sql = context.execute(sql).unwrap().collect().unwrap();

    let df_pl = df
        .select(&[
            col("a"),
            col("a").floor().alias("floor"),
            col("a").ceil().alias("ceil"),
        ])
        .collect()
        .unwrap();
    assert!(df_sql.equals_missing(&df_pl));
}

#[test]
#[cfg(feature = "csv")]
fn iss_8395() -> PolarsResult<()> {
    let mut context = SQLContext::new();
    let sql = r#"
    with foods as (
        SELECT *
        FROM read_csv('../../examples/datasets/foods1.csv')
    )
    select * from foods where category IN ('vegetables', 'seafood')"#;
    let res = context.execute(sql)?;
    let df = res.collect()?;

    // assert that the df only contains [vegetables, seafood]
    let s = df.column("category")?.unique()?.sort(Default::default())?;
    let expected = Column::new("category".into(), &["seafood", "vegetables"]);
    assert!(s.equals(&expected));
    Ok(())
}

#[test]
fn iss_8419() {
    let df = df! {
      "Year"=> [2018, 2018, 2019, 2019, 2020, 2020],
      "Country"=> ["US", "UK", "US", "UK", "US", "UK"],
      "Sales"=> [1000, 2000, 3000, 4000, 5000, 6000]
    }
    .unwrap()
    .lazy();

    let expected = df
        .clone()
        .select(&[
            col("Year"),
            col("Country"),
            col("Sales"),
            col("Sales")
                .sort(SortOptions::default().with_order_descending(true))
                .cum_sum(false)
                .alias("SalesCumulative"),
        ])
        .sort(["SalesCumulative"], Default::default())
        .collect()
        .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let query = r#"
    SELECT
        Year,
        Country,
        Sales,
        SUM(Sales) OVER (ORDER BY Sales DESC) as SalesCumulative
    FROM
        df
    ORDER BY
        SalesCumulative
    "#;
    let df = ctx.execute(query).unwrap().collect().unwrap();

    assert!(df.equals(&expected))
}

#[test]
fn iss_17381() {
    let users = df! {
        "id" => ["1", "2", "3", "4", "5", "6"],
        "email" => [
            "john.doe@company.com",
            "jane.doe@company.com",
            "doe.smith@company.com",
            "emily.johnsom@company.com",
            "alex.brown@company.com",
            "michael.davies@company.com",
        ]
    }
    .unwrap()
    .lazy();

    let user_groups = df! {
        "user_id" => ["1", "2", "3", "4", "5", "6"],
        "group_id" => ["1", "1", "2", "2", "3", "4"]
    }
    .unwrap()
    .lazy();

    let group_group = df! {
    "parent_id" => ["3", "3", "4"],
    "child_id" => ["1", "2", "3"]
    }
    .unwrap()
    .lazy();

    let deals = df! {
        "id" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "region" => ["East", "East", "East", "East", "East", "East", "West", "West", "West", "West", "West", "West"],
        "owner_id" => ["1", "1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "5"],
        "product" => ["Tee", "Golf", "Fancy", "Tee", "Golf", "Fancy", "Tee", "Golf", "Fancy", "Tee", "Golf", "Fancy"],
        "units" => [12, 12, 12, 10, 10, 10, 11, 11, 12, 12, 11, 11],
        "price" => [11.04, 13.00, 11.96, 11.27, 12.12, 13.74, 11.44, 12.63, 12.06, 13.42, 11.48, 11.48]
    }.unwrap().lazy();

    let mut ctx = SQLContext::new();

    ctx.register("users", users);
    ctx.register("user_groups", user_groups);
    ctx.register("group_group", group_group);
    ctx.register("deals", deals);

    let query = r#"
    WITH
      "user_by_email" AS (
        SELECT "users"."id"
          FROM "users"
          WHERE "email" IN ('john.doe@company.com')
      ),
      "user_child" AS (
        SELECT "user_groups"."user_id" AS "input_user_id", "group_group"."child_id"
          FROM "user_groups"
          INNER JOIN "user_by_email" ON "user_groups"."user_id" = "user_by_email"."id"
          INNER JOIN "group_group" ON "user_groups"."group_id" = "group_group"."parent_id"
      ),
      "deals_authz" AS (
        SELECT *
          FROM "deals"
          WHERE
          (
            ("deals"."owner_id" IN (SELECT "users"."id" FROM "users" WHERE "email" = 'john.doe@company.com'))
            OR
            ("deals"."owner_id" IN
              (SELECT DISTINCT "right"."user_id"
                FROM "user_groups" AS "left"
                  INNER JOIN "user_by_email" ON "user_groups"."user_id" = "user_by_email"."id"
                  INNER JOIN "user_groups" AS "right" ON "left"."group_id" = "right"."group_id"
            ))
          )
          OR ("deals"."owner_id" IN
            (SELECT DISTINCT "user_groups"."user_id"
              FROM "user_groups"
              INNER JOIN "user_child" ON "user_groups"."group_id" = "user_child"."child_id"
          ))
      )
    SELECT "id"
      FROM "deals_authz" AS "deals"
      ORDER BY "id" ASC
    "#;

    let _df = ctx.execute(query).unwrap().collect().unwrap();
    //dbg!(df);
}
