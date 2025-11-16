//! Helper utilities for SQL parsing and analysis
//!
//! This module contains Visitor implementations and helper functions for analyzing SQL AST nodes.

use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use sqlparser::ast::{Expr as SQLExpr, Select, SelectItem, Visit, Visitor};

/// Visitor that checks if an expression tree contains a reference to a specific table.
pub(crate) struct FindTableIdentifier<'a> {
    table_name: &'a str,
    found: bool,
}

impl<'a> FindTableIdentifier<'a> {
    pub(crate) fn new(table_name: &'a str) -> Self {
        Self {
            table_name,
            found: false,
        }
    }

    pub(crate) fn found(&self) -> bool {
        self.found
    }
}

impl<'a> Visitor for FindTableIdentifier<'a> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if let SQLExpr::CompoundIdentifier(idents) = expr {
            if idents.len() >= 2 && idents[0].value.as_str() == self.table_name {
                self.found = true; // return immediately on first match
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Visitor that checks if a SelectItem contains only literals/constants (no column references or subqueries).
/// This is used to detect the edge case where "SELECT 1, 2" should broadcast to table height.
/// Any identifier (simple or compound) indicates a column/table reference, making it non-literal.
pub(crate) struct LiteralOnlyVisitor {
    has_non_literal: bool,
}

impl LiteralOnlyVisitor {
    pub(crate) fn new() -> Self {
        Self {
            has_non_literal: false,
        }
    }

    pub(crate) fn has_non_literal(&self) -> bool {
        self.has_non_literal
    }
}

impl Visitor for LiteralOnlyVisitor {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        match expr {
            // Any identifier means we're referencing columns/tables, not just literals
            SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) => {
                self.has_non_literal = true;
                ControlFlow::Break(())
            },
            // Continue recursively for all other expressions
            _ => ControlFlow::Continue(()),
        }
    }
}

/// Check if a SELECT statement contains only literals/constants (no column references or subqueries).
/// Returns true if all projections are literal-only expressions.
pub(crate) fn is_literal_only_select(select_stmt: &Select) -> bool {
    if select_stmt.projection.is_empty() {
        return false;
    }

    for item in &select_stmt.projection {
        let mut visitor = LiteralOnlyVisitor::new();

        match item {
            SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                let _ = expr.visit(&mut visitor);
                if visitor.has_non_literal() {
                    return false;
                }
            },
            // Wildcards are not literals
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => return false,
        }
    }

    true
}

/// Check if a SQL expression contains a reference to a specific table.
pub(crate) fn expr_refers_to_table(expr: &SQLExpr, table_name: &str) -> bool {
    let mut table_finder = FindTableIdentifier::new(table_name);
    let _ = expr.visit(&mut table_finder);
    table_finder.found()
}

/// Check if all columns referred to in a Polars expression exist in the given Schema.
pub(crate) fn expr_cols_all_in_schema(expr: &Expr, schema: &Schema) -> bool {
    let mut found_cols = false;
    let mut all_in_schema = true;
    for e in expr.into_iter() {
        if let Expr::Column(name) = e {
            found_cols = true;
            if !schema.contains(name.as_str()) {
                all_in_schema = false;
                break;
            }
        }
    }
    found_cols && all_in_schema
}
