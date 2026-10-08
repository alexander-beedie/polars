use polars::frame::row::{AnyValueBuffer, Row, rows_to_schema_supertypes, rows_to_supertypes};
use polars::prelude::*;
use pyo3::exceptions::{PyException, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyMapping, PyString, PyTuple};

use super::PyDataFrame;
use crate::conversion::Wrap;
#[cfg(feature = "dtype-map")]
use crate::conversion::any_value::py_values_to_map_series;
use crate::conversion::any_value::{py_object_to_any_value, py_series_of};
use crate::error::PyPolarsErr;
use crate::interop;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    #[pyo3(signature = (data, schema=None, strict=true, infer_schema_length=None))]
    pub fn from_rows(
        py: Python<'_>,
        data: Vec<Bound<PyAny>>,
        schema: Option<Wrap<Schema>>,
        strict: bool,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let schema = schema.map(|wrap| wrap.0);
        let dtypes: Vec<&DataType> = schema.iter().flat_map(Schema::iter_values).collect();
        let mut batched: Vec<Option<Vec<Bound<PyAny>>>> = dtypes
            .iter()
            .map(|dtype| is_batched(dtype).then(Vec::new))
            .collect();
        let data = data
            .iter()
            .map(|row| {
                // lists and tuples are read in place, any other sequence through a `Vec`
                if let Ok(tuple) = row.cast::<PyTuple>() {
                    read_row(tuple.iter(), &dtypes, &mut batched, strict)
                } else if let Ok(list) = row.cast::<PyList>() {
                    read_row(list.iter(), &dtypes, &mut batched, strict)
                } else {
                    let values = row.extract::<Vec<Bound<PyAny>>>()?;
                    read_row(values.into_iter(), &dtypes, &mut batched, strict)
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let batched = batched
            .into_iter()
            .zip(&dtypes)
            .map(|(cells, dtype)| {
                cells
                    .map(|cells| batched_column(py, &cells, dtype, strict))
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.enter_polars(move || {
            finish_from_rows(data, schema, batched, strict, infer_schema_length)
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, schema=None, schema_overrides=None, strict=true, infer_schema_length=None))]
    pub fn from_dicts(
        py: Python<'_>,
        data: &Bound<PyAny>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        strict: bool,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let schema = schema.map(|wrap| wrap.0);
        let schema_overrides = schema_overrides.map(|wrap| wrap.0);
        let dtype_hint = |name: &str| {
            schema_overrides
                .as_ref()
                .and_then(|s| s.get(name))
                .or_else(|| schema.as_ref().and_then(|s| s.get(name)))
                .cloned()
        };

        // the leading records infer the schema (none are needed if it is complete)
        let schema_is_complete = schema.as_ref().is_some_and(|s| {
            s.iter_names()
                .all(|name| dtype_hint(name).is_some_and(|dtype| dtype.is_known()))
        });
        let n_infer = match infer_schema_length {
            _ if schema_is_complete => 0,
            Some(n) => n.max(1),
            None => usize::MAX,
        };
        let mut records = data.try_iter()?;
        let leading = records
            .by_ref()
            .take(n_infer)
            .map(|record| Record::new(record?))
            .collect::<PyResult<Vec<_>>>()?;

        let names: Vec<String> = match &schema {
            Some(schema) => schema.iter_names().map(|name| name.to_string()).collect(),
            None => {
                // in order of appearance
                let mut names = PlIndexSet::default();
                for record in &leading {
                    record.add_names(&mut names)?;
                }
                names.into_iter().collect()
            },
        };
        let keys: Vec<Bound<PyString>> = names
            .iter()
            .map(|name| PyString::intern(py, name))
            .collect();
        let hints: Vec<Option<DataType>> = names.iter().map(|name| dtype_hint(name)).collect();
        let read = |record: &Record, i: usize| record.value(&keys[i], hints[i].as_ref(), strict);
        let mut batched: Vec<Option<Vec<Bound<PyAny>>>> = hints
            .iter()
            .map(|dtype| dtype.as_ref().is_some_and(is_batched).then(Vec::new))
            .collect();
        let mut rows = Vec::with_capacity(leading.len());
        for record in &leading {
            let mut row = Vec::with_capacity(names.len());
            for (i, cells) in batched.iter_mut().enumerate() {
                row.push(match cells {
                    Some(cells) => {
                        cells.push(record.cell(&keys[i])?);
                        AnyValue::Null
                    },
                    None => read(record, i)?,
                });
            }
            rows.push(Row(row));
        }

        let mut schema = schema
            .unwrap_or_else(|| columns_names_to_empty_schema(names.iter().map(String::as_str)));
        resolve_schema_overrides(&mut schema, schema_overrides);
        update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;

        // values move into the buffers; if strict, a rejected value is read again, for
        // the error, otherwise it becomes null
        let capacity = data.len()?;
        let mut buffers: Vec<AnyValueBuffer> = schema
            .iter_values()
            .map(|dtype| AnyValueBuffer::new(dtype, capacity))
            .collect();
        let push = |buffer: &mut AnyValueBuffer<'static>, value, record: &Record, i| {
            if !strict {
                buffer.add_or_null(value);
            } else if buffer.add(value, true).is_none() {
                buffer
                    .add_fallible(&read(record, i)?, true)
                    .map_err(PyPolarsErr::from)?;
            }
            PyResult::Ok(())
        };
        let mut height = leading.len();
        for (record, row) in leading.iter().zip(rows) {
            for (i, (buffer, value)) in buffers.iter_mut().zip(row.0).enumerate() {
                if batched[i].is_none() {
                    push(buffer, value, record, i)?;
                }
            }
        }
        for record in records {
            let record = Record::new(record?)?;
            for (i, (buffer, cells)) in buffers.iter_mut().zip(&mut batched).enumerate() {
                match cells {
                    Some(cells) => cells.push(record.cell(&keys[i])?),
                    None => push(buffer, read(&record, i)?, &record, i)?,
                }
            }
            height += 1;
        }
        let batched = batched
            .into_iter()
            .zip(schema.iter_values())
            .map(|(cells, dtype)| {
                cells
                    .map(|cells| batched_column(py, &cells, dtype, strict))
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;

        py.enter_polars_df(move || {
            let columns = buffers
                .into_iter()
                .zip(batched)
                .zip(schema.iter_names())
                .map(|((buffer, batched), name)| {
                    let s = match batched {
                        Some(s) => s,
                        None => buffer.into_series()?,
                    };
                    Ok(s.with_name(name.clone()).into())
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            DataFrame::new(height, columns)
        })
    }

    #[staticmethod]
    pub fn from_arrow_record_batches(
        py: Python<'_>,
        rb: Vec<Bound<PyAny>>,
        schema: Bound<PyAny>,
    ) -> PyResult<Self> {
        let df = interop::arrow::to_rust::to_rust_df(py, &rb, schema)?;
        Ok(Self::from(df))
    }
}

/// Read a row's values with their column dtypes, as a dict can be a Struct or a Map.
///
/// The values of a `batched` column are kept instead, with a null in the row.
fn read_row<'py>(
    values: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
    dtypes: &[&DataType],
    batched: &mut [Option<Vec<Bound<'py, PyAny>>>],
    strict: bool,
) -> PyResult<Row<'static>> {
    let mut row = Vec::with_capacity(values.len());
    for (i, value) in values.enumerate() {
        row.push(match batched.get_mut(i) {
            Some(Some(cells)) => {
                cells.push(value);
                AnyValue::Null
            },
            _ => py_object_to_any_value(&value, strict, true, dtypes.get(i).copied())?,
        });
    }
    Ok(Row(row))
}

/// Whether the values of a column of this dtype are converted all at once, at the end: a
/// Map converts its keys and values with the Python constructor, which is slow to call for
/// every value.
fn is_batched(dtype: &DataType) -> bool {
    dtype.is_known() && dtype.contains_map()
}

/// Convert the values of a batched column all at once or, if that fails, one by one through
/// the row buffers, which then give the result (or error).
fn batched_column(
    py: Python<'_>,
    cells: &[Bound<'_, PyAny>],
    dtype: &DataType,
    strict: bool,
) -> PyResult<Series> {
    let one_by_one = |cells: &[Bound<'_, PyAny>]| -> PyResult<Series> {
        let mut buffer = AnyValueBuffer::new(dtype, cells.len());
        for cell in cells {
            let value = if cell.is_none() {
                AnyValue::Null
            } else {
                py_object_to_any_value(cell, strict, true, Some(dtype))?
            };
            if strict {
                buffer
                    .add_fallible(&value, true)
                    .map_err(PyPolarsErr::from)?;
            } else {
                buffer.add_or_null(value);
            }
        }
        Ok(buffer.into_series().map_err(PyPolarsErr::from)?)
    };
    match dtype {
        #[cfg(feature = "dtype-map")]
        DataType::Map(..) => py_values_to_map_series(py, cells, dtype, strict, one_by_one),
        _ => match py_series_of(
            py,
            PlSmallStr::EMPTY,
            PyList::new(py, cells)?,
            dtype,
            strict,
        ) {
            // not an `Exception`, e.g. a `KeyboardInterrupt`
            Err(err) if !err.is_instance_of::<PyException>(py) => Err(err),
            Err(_) => one_by_one(cells),
            s => s,
        },
    }
}

fn finish_from_rows(
    rows: Vec<Row>,
    schema: Option<Schema>,
    batched: Vec<Option<Series>>,
    strict: bool,
    infer_schema_length: Option<usize>,
) -> PyResult<PyDataFrame> {
    let schema = if let Some(mut schema) = schema {
        update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;
        schema
    } else {
        rows_to_schema_supertypes(&rows, infer_schema_length).map_err(PyPolarsErr::from)?
    };

    // the rows hold nulls in place of the batched columns
    let mut rows_schema = schema.clone();
    for (dtype, s) in rows_schema.iter_values_mut().zip(&batched) {
        if s.is_some() {
            *dtype = DataType::Null;
        }
    }
    let mut df =
        DataFrame::from_rows_and_schema(&rows, &rows_schema, strict).map_err(PyPolarsErr::from)?;
    for (i, (s, dtype)) in batched.into_iter().zip(schema.iter_values()).enumerate() {
        let Some(s) = s else { continue };
        let name = df.columns()[i].name().clone();
        // rows narrower than the schema have no values for its last columns
        let s = if s.is_empty() {
            Series::full_null(name, df.height(), dtype)
        } else {
            s.with_name(name)
        };
        df.replace_column(i, s.into()).map_err(PyPolarsErr::from)?;
    }
    Ok(df.into())
}

fn update_schema_from_rows(
    schema: &mut Schema,
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PyResult<()> {
    let schema_is_complete = schema.iter_values().all(|dtype| dtype.is_known());
    if schema_is_complete {
        return Ok(());
    }

    // TODO: Only infer dtypes for columns with an unknown dtype
    let inferred_dtypes =
        rows_to_supertypes(rows, infer_schema_length).map_err(PyPolarsErr::from)?;
    let inferred_dtypes_slice = inferred_dtypes.as_slice();

    for (i, dtype) in schema.iter_values_mut().enumerate() {
        if !dtype.is_known() {
            *dtype = inferred_dtypes_slice.get(i).ok_or_else(|| {
                polars_err!(SchemaMismatch: "the number of columns in the schema does not match the data")
            })
            .map_err(PyPolarsErr::from)?
            .clone();
        }
    }
    Ok(())
}

/// Override the data type of certain schema fields.
///
/// Overrides for nonexistent columns are ignored.
fn resolve_schema_overrides(schema: &mut Schema, schema_overrides: Option<Schema>) {
    if let Some(overrides) = schema_overrides {
        for (name, dtype) in overrides.into_iter() {
            schema.set_dtype(name.as_str(), dtype);
        }
    }
}

fn columns_names_to_empty_schema<'a, I>(column_names: I) -> Schema
where
    I: IntoIterator<Item = &'a str>,
{
    let fields = column_names
        .into_iter()
        .map(|c| Field::new(c.into(), DataType::Unknown(Default::default())));
    Schema::from_iter(fields)
}

/// A dict (or, slower, any mapping) record; `None` is a record of nulls.
enum Record<'py> {
    Null,
    Dict(Bound<'py, PyDict>),
    Mapping(Bound<'py, PyMapping>),
}

impl<'py> Record<'py> {
    fn new(record: Bound<'py, PyAny>) -> PyResult<Self> {
        if record.is_none() {
            return Ok(Self::Null);
        }
        Ok(match record.cast_into::<PyDict>() {
            Ok(dict) => Self::Dict(dict),
            Err(err) => Self::Mapping(err.into_inner().cast_into::<PyMapping>()?),
        })
    }

    /// Add the record's keys to the (ordered) schema names.
    fn add_names(&self, names: &mut PlIndexSet<String>) -> PyResult<()> {
        let keys = match self {
            Self::Null => return Ok(()),
            Self::Dict(dict) => dict.keys(),
            Self::Mapping(mapping) => mapping.keys()?,
        };
        for key in keys {
            let key = key.cast::<PyString>()?.to_str()?;
            if !names.contains(key) {
                names.insert(key.to_owned());
            }
        }
        Ok(())
    }

    #[inline]
    fn get(&self, key: &Bound<'py, PyString>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(match self {
            Self::Null => None,
            Self::Dict(dict) => dict.get_item(key)?,
            Self::Mapping(mapping) => match mapping.get_item(key) {
                Err(err) if err.is_instance_of::<PyKeyError>(mapping.py()) => None,
                value => Some(value?),
            },
        })
    }

    #[inline]
    fn value(
        &self,
        key: &Bound<'py, PyString>,
        dtype: Option<&DataType>,
        strict: bool,
    ) -> PyResult<AnyValue<'static>> {
        match self.get(key)? {
            Some(value) if !value.is_none() => py_object_to_any_value(&value, strict, true, dtype),
            _ => Ok(AnyValue::Null),
        }
    }

    /// The value at `key` as a Python object, `None` if missing.
    fn cell(&self, key: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self
            .get(key)?
            .unwrap_or_else(|| key.py().None().into_bound(key.py())))
    }
}
