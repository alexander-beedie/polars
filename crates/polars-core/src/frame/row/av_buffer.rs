#![deny(unsafe_op_in_unsafe_fn)]

use std::hint::unreachable_unchecked;

use polars_arrow::bitmap::BitmapBuilder;
#[cfg(feature = "dtype-decimal")]
use polars_compute::decimal::DecimalFmtBuffer;
#[cfg(feature = "dtype-struct")]
use polars_utils::pl_str::PlSmallStr;

use super::*;
use crate::chunked_array::builder::NullChunkedBuilder;
#[cfg(feature = "dtype-struct")]
use crate::prelude::any_value::arr_to_any_value;

#[derive(Clone)]
pub enum AnyValueBuffer<'a> {
    Boolean(BooleanChunkedBuilder),
    #[cfg(feature = "dtype-i8")]
    Int8(PrimitiveChunkedBuilder<Int8Type>),
    #[cfg(feature = "dtype-i16")]
    Int16(PrimitiveChunkedBuilder<Int16Type>),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    #[cfg(feature = "dtype-u8")]
    UInt8(PrimitiveChunkedBuilder<UInt8Type>),
    #[cfg(feature = "dtype-u16")]
    UInt16(PrimitiveChunkedBuilder<UInt16Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    #[cfg(feature = "dtype-date")]
    Date(PrimitiveChunkedBuilder<Int32Type>),
    #[cfg(feature = "dtype-datetime")]
    Datetime(
        PrimitiveChunkedBuilder<Int64Type>,
        TimeUnit,
        Option<TimeZone>,
    ),
    #[cfg(feature = "dtype-duration")]
    Duration(PrimitiveChunkedBuilder<Int64Type>, TimeUnit),
    #[cfg(feature = "dtype-time")]
    Time(PrimitiveChunkedBuilder<Int64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    String(StringChunkedBuilder),
    Null(NullChunkedBuilder),
    All(DataType, Vec<AnyValue<'a>>),
}

impl<'a> AnyValueBuffer<'a> {
    #[inline]
    pub fn add(&mut self, val: AnyValue<'_>, strict: bool) -> Option<()> {
        use AnyValueBuffer::*;
        if strict && !is_lossless(self, &val) {
            return None;
        }
        match (self, val) {
            (Boolean(builder), AnyValue::Null) => builder.append_null(),
            (Boolean(builder), AnyValue::Boolean(v)) => builder.append_value(v),
            // as when cast, a number is true if nonzero (other values cannot be cast)
            (Boolean(builder), val) if val.dtype().is_numeric() => {
                builder.append_value(val.extract::<f64>()? != 0.0)
            },
            (Int32(builder), AnyValue::Null) => builder.append_null(),
            (Int32(builder), val) => builder.append_value(val.extract()?),
            (Int64(builder), AnyValue::Null) => builder.append_null(),
            (Int64(builder), val) => builder.append_value(val.extract()?),
            (UInt32(builder), AnyValue::Null) => builder.append_null(),
            (UInt32(builder), val) => builder.append_value(val.extract()?),
            (UInt64(builder), AnyValue::Null) => builder.append_null(),
            (UInt64(builder), val) => builder.append_value(val.extract()?),
            (Float32(builder), AnyValue::Null) => builder.append_null(),
            (Float64(builder), AnyValue::Null) => builder.append_null(),
            (Float32(builder), val) => builder.append_value(val.extract()?),
            (Float64(builder), val) => builder.append_value(val.extract()?),
            (String(builder), AnyValue::String(v)) => builder.append_value(v),
            (String(builder), AnyValue::StringOwned(v)) => builder.append_value(v.as_str()),
            (String(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-i8")]
            (Int8(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-i8")]
            (Int8(builder), val) => builder.append_value(val.extract()?),
            #[cfg(feature = "dtype-i16")]
            (Int16(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-i16")]
            (Int16(builder), val) => builder.append_value(val.extract()?),
            #[cfg(feature = "dtype-u8")]
            (UInt8(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-u8")]
            (UInt8(builder), val) => builder.append_value(val.extract()?),
            #[cfg(feature = "dtype-u16")]
            (UInt16(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-u16")]
            (UInt16(builder), val) => builder.append_value(val.extract()?),
            #[cfg(feature = "dtype-date")]
            (Date(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-date")]
            (Date(builder), AnyValue::Date(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-date")]
            (Date(builder), val) if val.is_primitive_numeric() => {
                builder.append_value(val.extract()?)
            },
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder, _, _), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-datetime")]
            (
                Datetime(builder, tu_l, _),
                AnyValue::Datetime(v, tu_r, _) | AnyValue::DatetimeOwned(v, tu_r, _),
            ) => {
                // we convert right tu to left tu
                // so we swap.
                let v = crate::datatypes::time_unit::convert_time_units(v, tu_r, *tu_l);
                builder.append_value(v)
            },
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder, _, _), val) if val.is_primitive_numeric() => {
                builder.append_value(val.extract()?)
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(builder, _), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-duration")]
            (Duration(builder, tu_l), AnyValue::Duration(v, tu_r)) => {
                let v = crate::datatypes::time_unit::convert_time_units(v, tu_r, *tu_l);
                builder.append_value(v)
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(builder, _), val) if val.is_primitive_numeric() => {
                builder.append_value(val.extract()?)
            },
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Time(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-time")]
            (Time(builder), val) if val.is_primitive_numeric() => {
                builder.append_value(val.extract()?)
            },
            (Null(builder), AnyValue::Null) => builder.append_null(),
            // Struct and List can be recursive so use AnyValues for that; if strict,
            // reject values of the wrong kind (which would otherwise become null)
            (All(dtype, _), v) if strict && !is_nested_kind(dtype, &v) => return None,
            (All(_, vals), v) => vals.push(v.into_static()),

            // dynamic types
            (String(builder), av) => match av {
                AnyValue::Int64(v) => builder.append_value(format!("{v}")),
                AnyValue::Float64(v) => builder.append_value(format!("{v}")),
                AnyValue::Boolean(true) => builder.append_value("true"),
                AnyValue::Boolean(false) => builder.append_value("false"),
                #[cfg(feature = "dtype-decimal")]
                AnyValue::Decimal(v, _p, s) => {
                    let mut fmt = DecimalFmtBuffer::new();
                    builder.append_value(fmt.format_dec128(v, s, false, false));
                },
                _ => return None,
            },
            _ => return None,
        };
        Some(())
    }

    pub fn add_fallible(&mut self, val: &AnyValue<'a>, strict: bool) -> PolarsResult<()> {
        self.add(val.as_borrowed(), strict).ok_or_else(|| {
            polars_err!(
                ComputeError: "could not append value: {} of type: {} to the builder; make sure that all rows \
                have the same schema or consider increasing `infer_schema_length`\n\
                \n\
                it might also be that a value overflows the data-type's capacity", val, val.dtype()
            )
        })
    }

    /// Add `val`, or a null if it cannot be converted.
    #[inline]
    pub fn add_or_null(&mut self, val: AnyValue<'_>) {
        if self.add(val, false).is_none() {
            self.add(AnyValue::Null, false);
        }
    }

    pub fn reset(&mut self, capacity: usize, strict: bool) -> PolarsResult<Series> {
        use AnyValueBuffer::*;
        let out = match self {
            Boolean(b) => {
                let mut new = BooleanChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Int32(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Int64(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            UInt32(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            UInt64(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-date")]
            Date(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_date().into_series()
            },
            #[cfg(feature = "dtype-datetime")]
            Datetime(b, tu, tz) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                let tz = if capacity > 0 {
                    tz.clone()
                } else {
                    std::mem::take(tz)
                };
                new.finish().into_datetime(*tu, tz).into_series()
            },
            #[cfg(feature = "dtype-duration")]
            Duration(b, tu) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_duration(*tu).into_series()
            },
            #[cfg(feature = "dtype-time")]
            Time(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_time().into_series()
            },
            Float32(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Float64(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            String(b) => {
                let mut new = StringChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-i8")]
            Int8(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-i16")]
            Int16(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-u8")]
            UInt8(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-u16")]
            UInt16(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Null(b) => {
                let mut new = NullChunkedBuilder::new(b.field.name().clone(), 0);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            All(dtype, vals) => {
                let out =
                    Series::from_any_values_and_dtype(PlSmallStr::EMPTY, vals, dtype, strict)?;
                let mut new = Vec::with_capacity(capacity);
                std::mem::swap(&mut new, vals);
                out
            },
        };
        Ok(out)
    }

    pub fn into_series(mut self) -> PolarsResult<Series> {
        self.reset(0, false)
    }

    pub fn new(dtype: &DataType, capacity: usize) -> AnyValueBuffer<'a> {
        (dtype, capacity).into()
    }
}

/// Whether a value of nested `dtype` can be built from `av`, rather than becoming null.
fn is_nested_kind(dtype: &DataType, av: &AnyValue) -> bool {
    match dtype {
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fields) => match av {
            AnyValue::Null | AnyValue::Struct(..) | AnyValue::StructOwned(_) => true,
            AnyValue::List(s) => s.len() == fields.len(),
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(s, _) => s.len() == fields.len(),
            _ => false,
        },
        DataType::List(_) => matches!(av, AnyValue::Null | AnyValue::List(_)),
        #[cfg(feature = "dtype-map")]
        DataType::Map(..) => matches!(av, AnyValue::Null | AnyValue::Map(_) | AnyValue::List(_)),
        #[cfg(feature = "dtype-array")]
        DataType::Array(..) => {
            matches!(av, AnyValue::Null | AnyValue::List(_) | AnyValue::Array(..))
        },
        _ => true,
    }
}

/// Whether `av` can be added to a flat `buffer` without losing information, e.g. by
/// truncating 1.5 to an integer (values that cannot be converted fail to be added regardless).
fn is_lossless(buffer: &AnyValueBuffer, av: &AnyValue) -> bool {
    use AnyValueBuffer::*;
    match buffer {
        Boolean(_) => match av {
            AnyValue::Null | AnyValue::Boolean(_) => true,
            av => av.extract::<f64>().is_some_and(|v| v == 0.0 || v == 1.0),
        },
        Int32(_) | Int64(_) | UInt32(_) | UInt64(_) => is_integral(av),
        #[cfg(feature = "dtype-i8")]
        Int8(_) => is_integral(av),
        #[cfg(feature = "dtype-i16")]
        Int16(_) => is_integral(av),
        #[cfg(feature = "dtype-u8")]
        UInt8(_) => is_integral(av),
        #[cfg(feature = "dtype-u16")]
        UInt16(_) => is_integral(av),
        #[cfg(feature = "dtype-date")]
        Date(_) => is_integral(av),
        #[cfg(feature = "dtype-datetime")]
        Datetime(..) => is_integral(av),
        #[cfg(feature = "dtype-duration")]
        Duration(..) => is_integral(av),
        #[cfg(feature = "dtype-time")]
        Time(_) => is_integral(av),
        // floats narrow to the nearest value, and strings are formatted exactly
        Float32(_) | Float64(_) | String(_) | Null(_) | All(..) => true,
    }
}

/// Whether `av` has no fractional part that converting it to an integer would drop.
fn is_integral(av: &AnyValue) -> bool {
    match av {
        AnyValue::String(s) => s.parse::<i128>().is_ok(),
        AnyValue::StringOwned(s) => s.parse::<i128>().is_ok(),
        #[cfg(feature = "dtype-decimal")]
        AnyValue::Decimal(v, _, scale) => v % 10_i128.pow(*scale as u32) == 0,
        av => !av.is_float() || av.extract::<f64>().is_some_and(|v| v.fract() == 0.0),
    }
}

// datatype and length
impl From<(&DataType, usize)> for AnyValueBuffer<'_> {
    fn from(a: (&DataType, usize)) -> Self {
        let (dt, len) = a;
        use DataType::*;
        match dt {
            Boolean => AnyValueBuffer::Boolean(BooleanChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            Int32 => AnyValueBuffer::Int32(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            Int64 => AnyValueBuffer::Int64(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            UInt32 => AnyValueBuffer::UInt32(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            UInt64 => AnyValueBuffer::UInt64(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            #[cfg(feature = "dtype-i8")]
            Int8 => AnyValueBuffer::Int8(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            #[cfg(feature = "dtype-i16")]
            Int16 => AnyValueBuffer::Int16(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            #[cfg(feature = "dtype-u8")]
            UInt8 => AnyValueBuffer::UInt8(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            #[cfg(feature = "dtype-u16")]
            UInt16 => AnyValueBuffer::UInt16(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            #[cfg(feature = "dtype-date")]
            Date => AnyValueBuffer::Date(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => AnyValueBuffer::Datetime(
                PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len),
                *tu,
                tz.clone(),
            ),
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => {
                AnyValueBuffer::Duration(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len), *tu)
            },
            #[cfg(feature = "dtype-time")]
            Time => AnyValueBuffer::Time(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            Float32 => {
                AnyValueBuffer::Float32(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            Float64 => {
                AnyValueBuffer::Float64(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            String => AnyValueBuffer::String(StringChunkedBuilder::new(PlSmallStr::EMPTY, len)),
            Null => AnyValueBuffer::Null(NullChunkedBuilder::new(PlSmallStr::EMPTY, 0)),
            // Struct and List can be recursive so use AnyValues for that
            dt => AnyValueBuffer::All(dt.clone(), Vec::with_capacity(len)),
        }
    }
}

/// An [`AnyValueBuffer`] that should be used when we trust the builder
#[derive(Clone)]
pub enum AnyValueBufferTrusted<'a> {
    Boolean(BooleanChunkedBuilder),
    #[cfg(feature = "dtype-i8")]
    Int8(PrimitiveChunkedBuilder<Int8Type>),
    #[cfg(feature = "dtype-i16")]
    Int16(PrimitiveChunkedBuilder<Int16Type>),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    #[cfg(feature = "dtype-u8")]
    UInt8(PrimitiveChunkedBuilder<UInt8Type>),
    #[cfg(feature = "dtype-u16")]
    UInt16(PrimitiveChunkedBuilder<UInt16Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    String(StringChunkedBuilder),
    #[cfg(feature = "dtype-struct")]
    // not the trusted variant!
    Struct(BitmapBuilder, Vec<(AnyValueBuffer<'a>, PlSmallStr)>),
    Null(NullChunkedBuilder),
    All(DataType, Vec<AnyValue<'a>>),
}

impl<'a> AnyValueBufferTrusted<'a> {
    pub fn new(dtype: &DataType, len: usize) -> Self {
        (dtype, len).into()
    }

    #[inline]
    fn add_null(&mut self) {
        use AnyValueBufferTrusted::*;
        match self {
            Boolean(builder) => builder.append_null(),
            #[cfg(feature = "dtype-i8")]
            Int8(builder) => builder.append_null(),
            #[cfg(feature = "dtype-i16")]
            Int16(builder) => builder.append_null(),
            Int32(builder) => builder.append_null(),
            Int64(builder) => builder.append_null(),
            #[cfg(feature = "dtype-u8")]
            UInt8(builder) => builder.append_null(),
            #[cfg(feature = "dtype-u16")]
            UInt16(builder) => builder.append_null(),
            UInt32(builder) => builder.append_null(),
            UInt64(builder) => builder.append_null(),
            Float32(builder) => builder.append_null(),
            Float64(builder) => builder.append_null(),
            String(builder) => builder.append_null(),
            #[cfg(feature = "dtype-struct")]
            Struct(outer_validity, builders) => {
                outer_validity.push(false);
                for (b, _) in builders.iter_mut() {
                    b.add(AnyValue::Null, false);
                }
            },
            Null(builder) => builder.append_null(),
            All(_, vals) => vals.push(AnyValue::Null),
        }
    }

    /// # Safety
    /// The caller must ensure that the [`AnyValue`] type exactly matches the `Buffer` type.
    #[inline]
    unsafe fn add_physical(&mut self, val: &AnyValue<'_>) {
        // SAFETY: All unsafe blocks rely directly on the function contract.

        use AnyValueBufferTrusted::*;
        match self {
            Boolean(builder) => {
                let AnyValue::Boolean(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            #[cfg(feature = "dtype-i8")]
            Int8(builder) => {
                let AnyValue::Int8(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            #[cfg(feature = "dtype-i16")]
            Int16(builder) => {
                let AnyValue::Int16(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            Int32(builder) => {
                let AnyValue::Int32(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            Int64(builder) => {
                let AnyValue::Int64(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            #[cfg(feature = "dtype-u8")]
            UInt8(builder) => {
                let AnyValue::UInt8(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            #[cfg(feature = "dtype-u16")]
            UInt16(builder) => {
                let AnyValue::UInt16(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            UInt32(builder) => {
                let AnyValue::UInt32(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            UInt64(builder) => {
                let AnyValue::UInt64(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            Float32(builder) => {
                let AnyValue::Float32(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            Float64(builder) => {
                let AnyValue::Float64(v) = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_value(*v)
            },
            Null(builder) => {
                let AnyValue::Null = val else {
                    unsafe { unreachable_unchecked() }
                };
                builder.append_null()
            },
            _ => unreachable!(),
        }
    }

    /// Will add the [`AnyValue`] into [`Self`] and unpack as the physical type belonging to
    /// [`Self`]. This should only be used with physical buffers
    ///
    /// If a type is not primitive or String, the AnyValues will be converted to static
    ///
    /// # Safety
    /// The caller must ensure that the [`AnyValue`] type exactly matches the `Buffer` type and is
    /// owned.
    #[inline]
    pub unsafe fn add_unchecked_owned_physical(&mut self, val: &AnyValue<'a>) {
        use AnyValueBufferTrusted::*;
        match val {
            AnyValue::Null => self.add_null(),
            _ => {
                match self {
                    String(builder) => {
                        let AnyValue::StringOwned(v) = val else {
                            // SAFETY: Function contract.
                            unsafe { unreachable_unchecked() }
                        };
                        builder.append_value(v.as_str())
                    },
                    #[cfg(feature = "dtype-struct")]
                    Struct(outer_validity, builders) => {
                        let AnyValue::StructOwned(payload) = val else {
                            // SAFETY: Function contract.
                            unsafe { unreachable_unchecked() }
                        };
                        let avs = &*payload.0;

                        debug_assert_eq!(builders.len(), avs.len());
                        for ((builder, _), av) in builders.iter_mut().zip(avs.iter().cloned()) {
                            builder.add(av, false);
                        }
                        outer_validity.push(true);
                    },
                    All(_, vals) => vals.push(val.clone().into_static()),
                    // SAFETY: Function contract.
                    _ => unsafe { self.add_physical(val) },
                }
            },
        }
    }

    /// # Safety
    /// The caller must ensure that the [`AnyValue`] type exactly matches the `Buffer` type and is
    /// borrowed and if `val` is a `AnyValue::Struct` that the values are internally consistent.
    #[inline]
    pub unsafe fn add_unchecked_borrowed_physical(&mut self, val: &AnyValue<'a>) {
        use AnyValueBufferTrusted::*;
        match val {
            AnyValue::Null => self.add_null(),
            _ => {
                match self {
                    String(builder) => {
                        let AnyValue::String(v) = val else {
                            unsafe { unreachable_unchecked() }
                        };
                        builder.append_value(v)
                    },
                    #[cfg(feature = "dtype-struct")]
                    Struct(outer_validity, builders) => {
                        let AnyValue::Struct(idx, arr, fields) = *val else {
                            // SAFETY: Function contract.
                            unsafe { unreachable_unchecked() }
                        };
                        let arrays = arr.values();
                        debug_assert_eq!(builders.len(), arrays.len());
                        debug_assert_eq!(fields.len(), arrays.len());
                        for ((field, array), (builder, _)) in
                            fields.iter().zip(arrays).zip(builders.iter_mut())
                        {
                            // SAFETY: The values inside `val` need to be consistent, `idx` MUST be
                            // in-bounds for all `arr.values()` and `field.dtype` correct.
                            let av_new = unsafe { arr_to_any_value(&**array, idx, &field.dtype) };
                            builder.add(av_new, false);
                        }
                        outer_validity.push(true);
                    },
                    All(_, vals) => vals.push(val.clone().into_static()),
                    // SAFETY: Function contract.
                    _ => unsafe { self.add_physical(val) },
                }
            },
        }
    }

    /// Clear `self` and give `capacity`, returning the old contents as a [`Series`].
    pub fn reset(&mut self, capacity: usize, strict: bool) -> PolarsResult<Series> {
        use AnyValueBufferTrusted::*;
        let out = match self {
            Boolean(b) => {
                let mut new = BooleanChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Int32(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Int64(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            UInt32(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            UInt64(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Float32(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            Float64(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            String(b) => {
                let mut new = StringChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-i8")]
            Int8(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-i16")]
            Int16(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-u8")]
            UInt8(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-u16")]
            UInt16(b) => {
                let mut new = PrimitiveChunkedBuilder::new(b.field.name().clone(), capacity);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(outer_validity, b) => {
                // @Q? Maybe we need to add a length parameter here for ZFS's. I am not very happy
                // with just setting the length to zero for that case.
                if b.is_empty() {
                    return Ok(
                        StructChunked::from_series(PlSmallStr::EMPTY, 0, [].iter())?.into_series()
                    );
                }

                let mut min_len = usize::MAX;
                let mut max_len = usize::MIN;

                let v = b
                    .iter_mut()
                    .map(|(b, name)| {
                        let mut s = b.reset(capacity, strict)?;

                        min_len = min_len.min(s.len());
                        max_len = max_len.max(s.len());

                        s.rename(name.clone());
                        Ok(s)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let length = if min_len == 0 { 0 } else { max_len };

                let old_outer_validity = core::mem::take(outer_validity);
                outer_validity.reserve(capacity);

                StructChunked::from_series(PlSmallStr::EMPTY, length, v.iter())?
                    .with_outer_validity(Some(old_outer_validity.freeze()))
                    .into_series()
            },
            Null(b) => {
                let mut new = NullChunkedBuilder::new(b.field.name().clone(), 0);
                std::mem::swap(&mut new, b);
                new.finish().into_series()
            },
            All(dtype, vals) => {
                let mut swap_vals = Vec::with_capacity(capacity);
                std::mem::swap(vals, &mut swap_vals);
                Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &swap_vals, dtype, false)?
            },
        };

        Ok(out)
    }

    pub fn into_series(mut self) -> Series {
        // unwrap: non-strict does not error.
        self.reset(0, false).unwrap()
    }
}

impl From<(&DataType, usize)> for AnyValueBufferTrusted<'_> {
    fn from(a: (&DataType, usize)) -> Self {
        let (dt, len) = a;
        use DataType::*;
        match dt {
            Boolean => {
                AnyValueBufferTrusted::Boolean(BooleanChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            Int32 => {
                AnyValueBufferTrusted::Int32(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            Int64 => {
                AnyValueBufferTrusted::Int64(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            UInt32 => {
                AnyValueBufferTrusted::UInt32(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            UInt64 => {
                AnyValueBufferTrusted::UInt64(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            #[cfg(feature = "dtype-i8")]
            Int8 => {
                AnyValueBufferTrusted::Int8(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            #[cfg(feature = "dtype-i16")]
            Int16 => {
                AnyValueBufferTrusted::Int16(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            #[cfg(feature = "dtype-u8")]
            UInt8 => {
                AnyValueBufferTrusted::UInt8(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            #[cfg(feature = "dtype-u16")]
            UInt16 => {
                AnyValueBufferTrusted::UInt16(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            Float32 => {
                AnyValueBufferTrusted::Float32(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            Float64 => {
                AnyValueBufferTrusted::Float64(PrimitiveChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            String => {
                AnyValueBufferTrusted::String(StringChunkedBuilder::new(PlSmallStr::EMPTY, len))
            },
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                let outer_validity = BitmapBuilder::with_capacity(len);
                let buffers = fields
                    .iter()
                    .map(|field| {
                        let dtype = field.dtype().to_physical();
                        let buffer: AnyValueBuffer = (&dtype, len).into();
                        (buffer, field.name.clone())
                    })
                    .collect::<Vec<_>>();
                AnyValueBufferTrusted::Struct(outer_validity, buffers)
            },
            // List can be recursive so use AnyValues for that
            dt => AnyValueBufferTrusted::All(dt.clone(), Vec::with_capacity(len)),
        }
    }
}
