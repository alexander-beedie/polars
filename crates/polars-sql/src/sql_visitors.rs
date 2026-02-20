//! SQLVisitor helper implementations for traversing SQL AST expressions.
//!
//! This module provides visitor implementations used throughout the SQL interface
//! to analyze and check SQL expressions for various properties.

use std::borrow::Cow;
use std::ops::ControlFlow;

use polars_core::prelude::*;
use sqlparser::ast::{Expr as SQLExpr, ObjectName, Query, SetExpr, Visit, Visitor as SQLVisitor};
use sqlparser::keywords::ALL_KEYWORDS;

// ---------------------------------------------------------------------------
// ExprMatcher — generic boolean visitor for SQL expressions
// ---------------------------------------------------------------------------

/// Visitor that checks if any node in a SQL expression tree matches a predicate.
/// Uses early-exit (`ControlFlow::Break`) for efficiency.
struct ExprMatcher<F: Fn(&SQLExpr) -> bool> {
    predicate: F,
}

impl<F: Fn(&SQLExpr) -> bool> SQLVisitor for ExprMatcher<F> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if (self.predicate)(expr) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Check if any node in a SQL expression matches the given predicate.
fn expr_matches(expr: &SQLExpr, predicate: impl Fn(&SQLExpr) -> bool) -> bool {
    expr.visit(&mut ExprMatcher { predicate }).is_break()
}

/// Check if a SQL expression contains a reference to a specific table.
pub(crate) fn expr_refers_to_table(expr: &SQLExpr, table_name: &str) -> bool {
    expr_matches(
        expr,
        |e| matches!(e, SQLExpr::CompoundIdentifier(idents) if idents.len() >= 2 && idents[0].value.as_str() == table_name),
    )
}

/// Check if a SQL expression contains any table-qualified reference (e.g. `t1.col`).
pub(crate) fn expr_has_table_qualifier(expr: &SQLExpr) -> bool {
    expr_matches(
        expr,
        |e| matches!(e, SQLExpr::CompoundIdentifier(idents) if idents.len() >= 2),
    )
}

// ---------------------------------------------------------------------------
// QualifyExpression
// ---------------------------------------------------------------------------

/// Visitor used to check a SQL expression used in a QUALIFY clause.
/// (Confirms window functions are present and collects column refs in one pass).
pub(crate) struct QualifyExpression {
    has_window_functions: bool,
    column_refs: PlHashSet<String>,
}

impl QualifyExpression {
    fn new() -> Self {
        Self {
            has_window_functions: false,
            column_refs: PlHashSet::new(),
        }
    }

    pub(crate) fn analyze(expr: &SQLExpr) -> (bool, PlHashSet<String>) {
        let mut analyzer = Self::new();
        let _ = expr.visit(&mut analyzer);
        (analyzer.has_window_functions, analyzer.column_refs)
    }
}

impl SQLVisitor for QualifyExpression {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        match expr {
            SQLExpr::Function(func) if func.over.is_some() => {
                self.has_window_functions = true;
            },
            SQLExpr::Identifier(ident) => {
                self.column_refs.insert(ident.value.clone());
            },
            SQLExpr::CompoundIdentifier(idents) if !idents.is_empty() => {
                self.column_refs
                    .insert(idents.last().unwrap().value.clone());
            },
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// UnqualifiedColumnChecker
// ---------------------------------------------------------------------------

/// Check unqualified column references (i.e. `SQLExpr::Identifier`, not
/// `CompoundIdentifier`) in a SQL expression against two schemas.
///
/// Returns `(in_left, in_right)` indicating whether any unqualified identifier
/// was found in each schema. An identifier that exists in both schemas
/// contributes to both flags.
pub(crate) fn check_unqualified_columns(
    expr: &SQLExpr,
    left_schema: &Schema,
    right_schema: &Schema,
) -> (bool, bool) {
    struct Checker<'a> {
        left_schema: &'a Schema,
        right_schema: &'a Schema,
        in_left: bool,
        in_right: bool,
    }

    impl SQLVisitor for Checker<'_> {
        type Break = ();

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            if let SQLExpr::Identifier(ident) = expr {
                let name = ident.value.as_str();
                self.in_left |= self.left_schema.contains(name);
                self.in_right |= self.right_schema.contains(name);
            }
            ControlFlow::Continue(())
        }
    }

    let mut checker = Checker {
        left_schema,
        right_schema,
        in_left: false,
        in_right: false,
    };
    let _ = expr.visit(&mut checker);
    (checker.in_left, checker.in_right)
}

// ---------------------------------------------------------------------------
// RightTableColumnCollector
// ---------------------------------------------------------------------------

/// Walk a SQL expression and collect the names of columns that belong to the
/// right table. Uses SQL-level qualifiers (`right_table.col`) first; for
/// unqualified identifiers, falls back to schema membership (only in right
/// schema and not in left).
pub(crate) fn collect_right_table_column_names(
    sql_expr: &SQLExpr,
    right_table_name: &str,
    left_schema: &Schema,
    right_schema: &Schema,
) -> PlHashSet<String> {
    struct Collector<'a> {
        right_table_name: &'a str,
        left_schema: &'a Schema,
        right_schema: &'a Schema,
        right_cols: PlHashSet<String>,
    }

    impl SQLVisitor for Collector<'_> {
        type Break = ();

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            match expr {
                // Qualified: `right_table.col`
                SQLExpr::CompoundIdentifier(idents) if idents.len() >= 2 => {
                    if idents[0].value.as_str() == self.right_table_name {
                        let col_name = idents.last().unwrap().value.clone();
                        self.right_cols.insert(col_name);
                    }
                },
                // Unqualified: belongs to right if in right schema only
                SQLExpr::Identifier(ident) => {
                    let name = ident.value.as_str();
                    if self.right_schema.contains(name) && !self.left_schema.contains(name) {
                        self.right_cols.insert(name.to_owned());
                    }
                },
                _ => {},
            }
            ControlFlow::Continue(())
        }
    }

    let mut collector = Collector {
        right_table_name,
        left_schema,
        right_schema,
        right_cols: PlHashSet::new(),
    };
    let _ = sql_expr.visit(&mut collector);
    collector.right_cols
}

// ---------------------------------------------------------------------------
// AmbiguousColumnVisitor
// ---------------------------------------------------------------------------

/// Format an identifier, quoting only if necessary (or `force` is true).
fn maybe_quote(s: &str, force: bool) -> Cow<'_, str> {
    let needs_quoting = force
        || s.is_empty()
        || s.starts_with(|c: char| c.is_ascii_digit())
        || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || ALL_KEYWORDS.contains(&s.to_ascii_uppercase().as_str());
    if needs_quoting {
        Cow::Owned(format!("\"{s}\""))
    } else {
        Cow::Borrowed(s)
    }
}

/// Visitor that checks for unqualified references to columns that exist in
/// multiple tables (columns appearing in a USING clause are excluded from
/// the check as they are implicitly coalesced).
struct AmbiguousColumnVisitor<'a> {
    joined_aliases: &'a PlHashMap<String, PlHashMap<String, String>>,
    /// Original column names for each table that participated in a join.
    joined_table_columns: &'a PlHashMap<String, PlHashSet<String>>,
    using_cols: &'a PlHashSet<String>,
}

impl SQLVisitor for AmbiguousColumnVisitor<'_> {
    type Break = PolarsError;

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if let SQLExpr::Identifier(ident) = expr {
            let col = &ident.value;
            if self.using_cols.contains(col) {
                return ControlFlow::Continue(());
            }
            // A column is only worth checking if it appears in at least one
            // joined_aliases entry (i.e. was involved in a naming conflict).
            let in_any_alias = self
                .joined_aliases
                .values()
                .any(|cols| cols.contains_key(col));

            if in_any_alias {
                // Determine which tables actually contain this column
                let mut tables: Vec<_> = self
                    .joined_table_columns
                    .iter()
                    .filter_map(|(t, cols)| cols.contains(col).then_some(t.as_str()))
                    .collect();
                tables.sort();

                if tables.len() >= 2 {
                    let col_hint = maybe_quote(col, false);
                    let hints = tables
                        .iter()
                        .map(|t| format!("{}.{}", maybe_quote(t, false), col_hint));
                    return ControlFlow::Break(polars_err!(
                        SQLInterface: "ambiguous reference to column {} (use one of: {})",
                        maybe_quote(col, true), hints.collect::<Vec<_>>().join(", ")
                    ));
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Check a SQL expression for unqualified references to columns that
/// exist in multiple tables (columns appearing in a USING clause are
/// excluded from the check as they are implicitly coalesced).
pub(crate) fn check_for_ambiguous_column_refs(
    expr: &SQLExpr,
    joined_aliases: &PlHashMap<String, PlHashMap<String, String>>,
    joined_table_columns: &PlHashMap<String, PlHashSet<String>>,
    using_cols: &PlHashSet<String>,
) -> PolarsResult<()> {
    match expr.visit(&mut AmbiguousColumnVisitor {
        joined_aliases,
        joined_table_columns,
        using_cols,
    }) {
        ControlFlow::Break(err) => Err(err),
        ControlFlow::Continue(()) => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// TableIdentifierCollector
// ---------------------------------------------------------------------------

/// Visitor that collects all table identifiers referenced in a SQL query.
#[derive(Default)]
pub(crate) struct TableIdentifierCollector {
    pub(crate) tables: Vec<String>,
    pub(crate) include_schema: bool,
}

impl TableIdentifierCollector {
    pub(crate) fn collect_from_set_expr(&mut self, set_expr: &SetExpr) {
        // Recursively collect table identifiers from SetExpr nodes
        match set_expr {
            SetExpr::Table(tbl) => {
                self.tables.extend(if self.include_schema {
                    match (&tbl.schema_name, &tbl.table_name) {
                        (Some(schema), Some(table)) => Some(format!("{schema}.{table}")),
                        (None, Some(table)) => Some(table.clone()),
                        _ => None,
                    }
                } else {
                    tbl.table_name.clone()
                });
            },
            SetExpr::SetOperation { left, right, .. } => {
                self.collect_from_set_expr(left);
                self.collect_from_set_expr(right);
            },
            SetExpr::Query(query) => self.collect_from_set_expr(&query.body),
            _ => {},
        }
    }
}

impl SQLVisitor for TableIdentifierCollector {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
        // Collect from SetExpr nodes in the query body
        self.collect_from_set_expr(&query.body);
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
        // Table relation (eg: appearing in FROM clause)
        self.tables.extend(if self.include_schema {
            let parts: Vec<_> = relation
                .0
                .iter()
                .filter_map(|p| p.as_ident().map(|i| i.value.as_str()))
                .collect();
            (!parts.is_empty()).then(|| parts.join("."))
        } else {
            relation
                .0
                .last()
                .and_then(|p| p.as_ident())
                .map(|i| i.value.clone())
        });
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// WindowFunctionFinder
// ---------------------------------------------------------------------------

/// Check if a SQL expression contains explicit window functions.
pub(crate) fn expr_has_window_functions(expr: &SQLExpr) -> bool {
    expr_matches(
        expr,
        |e| matches!(e, SQLExpr::Function(f) if f.over.is_some()),
    )
}
