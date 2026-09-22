use std::sync::Arc;

use parking_lot::RwLock;
use polars::lazy::dsl::Expr;
use polars::sql::{
    FunctionRegistry, SQLContext, extract_table_identifiers, validate_function_name,
};
use polars_core::prelude::PlHashMap;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_plan::prelude::UserDefinedFunction;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyTuple};

use crate::error::PyPolarsErr;
use crate::expr::expr_to_py_plexpr;
use crate::utils::EnterPolarsExt;
use crate::{PyExpr, PyLazyFrame};

#[derive(Default)]
struct PythonFunctionRegistry {
    functions: RwLock<PlHashMap<String, Py<PyAny>>>,
}

impl FunctionRegistry for PythonFunctionRegistry {
    fn register(&mut self, _name: &str, _fun: UserDefinedFunction) -> PolarsResult<()> {
        polars_bail!(ComputeError: "native function registration is not supported")
    }

    fn get_udf(&self, _name: &str) -> PolarsResult<Option<UserDefinedFunction>> {
        Ok(None)
    }

    fn resolve_udf(&self, name: &str, args: Vec<Expr>) -> PolarsResult<Option<Expr>> {
        Python::attach(|py| {
            let function = {
                let functions = self.functions.read();
                let Some(function) = functions.get(name) else {
                    return Ok(None);
                };
                function.clone_ref(py)
            };
            let args = PyTuple::new(py, args.into_iter().map(|expr| expr_to_py_plexpr(py, expr)))
                .map_err(
                |err| polars_err!(ComputeError: "error preparing SQL function '{name}': {err}"),
            )?;
            let output = function.call1(py, args).map_err(
                |err| polars_err!(ComputeError: "error resolving SQL function '{name}': {err}"),
            )?;
            let pyexpr = output
                .getattr(py, "_pyexpr")
                .and_then(|output| output.extract::<PyExpr>(py).map_err(PyErr::from))
                .map_err(|_| {
                    polars_err!(ComputeError: "SQL function '{name}' must return a Polars Expr")
                })?;
            polars_plan::dsl::python_dsl::validate_python_udf_output_types(&pyexpr.inner)
                .map_err(|err| err.wrap_msg(|msg| format!("SQL function '{name}' {msg}")))?;
            Ok(Some(pyexpr.inner))
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.functions.read().contains_key(name)
    }

    fn is_empty(&self) -> bool {
        self.functions.read().is_empty()
    }
}

fn context_busy() -> PyPolarsErr {
    polars_err!(
        ComputeError:
        "SQLContext is already in use; SQL function callbacks cannot re-enter the same context"
    )
    .into()
}

#[pyclass(frozen, skip_from_py_object)]
pub struct PySQLContext {
    pub context: RwLock<SQLContext>,
    function_registry: Arc<PythonFunctionRegistry>,
}

#[pymethods]
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PySQLContext {
    #[staticmethod]
    #[allow(clippy::new_without_default)]
    pub fn new() -> PySQLContext {
        let function_registry: Arc<PythonFunctionRegistry> = Default::default();
        PySQLContext {
            context: RwLock::new(
                SQLContext::new().with_function_registry(function_registry.clone()),
            ),
            function_registry,
        }
    }

    /// Execute a SQL query in the current SQLContext.
    pub fn execute(&self, py: Python<'_>, query: &str) -> PyResult<PyLazyFrame> {
        py.enter_polars(|| {
            let mut context = self.context.try_write().ok_or_else(context_busy)?;
            context.execute(query).map_err(PyPolarsErr::from)
        })
        .map(Into::into)
    }

    /// Get a list of table names registered in the current SQLContext.
    pub fn get_tables(&self) -> PyResult<Vec<String>> {
        Ok(self
            .context
            .try_read()
            .ok_or_else(context_busy)?
            .get_tables())
    }

    /// Register a table in the current SQLContext.
    pub fn register(&self, name: &str, lf: PyLazyFrame) -> PyResult<()> {
        self.context
            .try_write()
            .ok_or_else(context_busy)?
            .register(name, lf.ldf.into_inner());
        Ok(())
    }

    /// Unregister a table from the current SQLContext.
    pub fn unregister(&self, name: &str) -> PyResult<()> {
        self.context
            .try_write()
            .ok_or_else(context_busy)?
            .unregister(name);
        Ok(())
    }

    /// Register a Python expression builder as a SQL function.
    pub fn register_function(
        &self,
        py: Python<'_>,
        name: &str,
        function: Py<PyAny>,
    ) -> PyResult<()> {
        if !function.bind(py).is_callable() {
            return Err(PyValueError::new_err("SQL function must be callable"));
        }
        let name =
            validate_function_name(name).map_err(|err| PyValueError::new_err(err.to_string()))?;

        let _context = self.context.try_read().ok_or_else(context_busy)?;
        let mut functions = self
            .function_registry
            .functions
            .try_write()
            .ok_or_else(context_busy)?;
        if functions.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "SQL function {name:?} is already registered; unregister it before registering a replacement"
            )));
        }
        functions.insert(name, function);
        Ok(())
    }

    /// Unregister a Python SQL function.
    pub fn unregister_function(&self, name: &str) -> PyResult<()> {
        let name = name.to_lowercase();
        let context = self.context.try_read().ok_or_else(context_busy)?;
        let mut functions = self
            .function_registry
            .functions
            .try_write()
            .ok_or_else(context_busy)?;
        let removed = functions.remove(&name);
        drop(functions);
        drop(context);
        drop(removed);
        Ok(())
    }

    /// Get the registered Python SQL function names.
    pub fn get_functions(&self) -> PyResult<Vec<String>> {
        let mut functions = self
            .function_registry
            .functions
            .try_read()
            .ok_or_else(context_busy)?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        functions.sort_unstable();
        Ok(functions)
    }

    /// Extract table identifiers from a SQL query string.
    #[staticmethod]
    #[pyo3(signature = (query, include_schema=true, unique=false))]
    pub fn table_identifiers(
        query: &str,
        include_schema: bool,
        unique: bool,
    ) -> PyResult<Vec<String>> {
        extract_table_identifiers(query, include_schema, unique)
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }
}
