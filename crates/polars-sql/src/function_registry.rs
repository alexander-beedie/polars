//! This module defines a FunctionRegistry for supported SQL functions and UDFs.

use polars_error::{PolarsResult, polars_bail};
use polars_plan::prelude::Expr;
pub use polars_plan::prelude::FunctionOptions;
use polars_plan::prelude::udf::UserDefinedFunction;

/// Validate and normalize a custom SQL function name.
pub fn validate_function_name(name: &str) -> PolarsResult<String> {
    let name = name.to_lowercase();
    let mut chars = name.chars();
    if !matches!(chars.next(), Some('_' | 'a'..='z'))
        || !chars.all(|c| matches!(c, '_' | 'a'..='z' | '0'..='9'))
    {
        polars_bail!(InvalidOperation: "invalid SQL function name: {name:?}")
    }
    if crate::functions::is_builtin_function(&name) {
        polars_bail!(InvalidOperation: "cannot register SQL function {name:?}: name conflicts with a built-in function")
    }
    let keyword = name.to_ascii_uppercase();
    if sqlparser::keywords::ALL_KEYWORDS
        .binary_search(&keyword.as_str())
        .is_ok()
    {
        polars_bail!(InvalidOperation: "cannot register SQL function {name:?}: name is a reserved SQL keyword")
    }
    Ok(name)
}

/// A registry that holds user defined functions.
pub trait FunctionRegistry: Send + Sync {
    /// Register a function.
    fn register(&mut self, name: &str, fun: UserDefinedFunction) -> PolarsResult<()>;
    /// Call a user defined function.
    fn get_udf(&self, name: &str) -> PolarsResult<Option<UserDefinedFunction>>;
    /// Resolve a user defined function call to an expression.
    fn resolve_udf(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Option<Expr>> {
        Ok(self.get_udf(name)?.map(|udf| udf.call(args)))
    }
    /// Whether the registry guarantees that it contains no functions.
    fn is_empty(&self) -> bool {
        false
    }
    /// Check if a function is registered.
    fn contains(&self, name: &str) -> bool;
}

/// A default registry that does not support registering or calling functions.
pub struct DefaultFunctionRegistry {}

impl FunctionRegistry for DefaultFunctionRegistry {
    fn register(&mut self, _name: &str, _fun: UserDefinedFunction) -> PolarsResult<()> {
        polars_bail!(ComputeError: "'register' not implemented on DefaultFunctionRegistry'")
    }

    fn get_udf(&self, _name: &str) -> PolarsResult<Option<UserDefinedFunction>> {
        polars_bail!(ComputeError: "'get_udf' not implemented on DefaultFunctionRegistry'")
    }
    fn is_empty(&self) -> bool {
        true
    }
    fn contains(&self, _name: &str) -> bool {
        false
    }
}
