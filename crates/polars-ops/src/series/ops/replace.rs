use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_defs::join::JoinValidation;
use polars_error::polars_ensure;

use crate::prelude::*;

fn find_output_length(
    fnname: &str,
    items: impl IntoIterator<Item = (&'static str, usize)>,
) -> PolarsResult<usize> {
    let mut length = 1;
    for (argument_idx, (argument, l)) in items.into_iter().enumerate() {
        if l != 1 {
            if l != length && length != 1 {
                polars_bail!(
                    length_mismatch = fnname,
                    l,
                    length,
                    argument = argument,
                    argument_idx = argument_idx
                );
            }
            length = l;
        }
    }
    Ok(length)
}

/// Replace values by different values of the same data type.
pub fn replace(s: &Series, old: &ListChunked, new: &ListChunked) -> PolarsResult<Series> {
    let (old, new) = unpack_replacements("replace", s, old, new, None)?;

    if old.is_empty() {
        return Ok(s.clone());
    }

    let dtype = s.dtype();
    execute_replacement(s, old, new, Some(dtype), CastOptions::Strict, Some(s))
}

/// Replace all values by different values.
///
/// Unmatched values are replaced by a default value.
pub fn replace_or_default(
    s: &Series,
    old: &ListChunked,
    new: &ListChunked,
    default: &Series,
    return_dtype: Option<DataType>,
) -> PolarsResult<Series> {
    let (old, new) = unpack_replacements("replace_strict", s, old, new, Some(default))?;

    polars_ensure!(
        default.len() == s.len() || default.len() == 1,
        InvalidOperation: "`default` input for `replace_strict` must have the same length as the input or have length 1"
    );

    let return_dtype = match return_dtype {
        Some(dtype) => dtype,
        None => try_get_supertype(new.dtype(), default.dtype())?,
    };
    execute_replacement(
        s,
        old,
        new,
        Some(&return_dtype),
        CastOptions::NonStrict,
        Some(default),
    )
}

/// Replace all values by different values.
///
/// Raises an error if not all values were replaced.
pub fn replace_strict(
    s: &Series,
    old: &ListChunked,
    new: &ListChunked,
    return_dtype: Option<DataType>,
) -> PolarsResult<Series> {
    let (old, new) = unpack_replacements("replace_strict", s, old, new, None)?;

    polars_ensure!(
        !old.is_empty() || s.len() == s.null_count(),
        InvalidOperation: "must specify which values to replace"
    );

    // Extra check because strict_cast is too permissive, e.g. allows string -> struct cast.
    if old.dtype().can_cast_to(s.dtype()) != Some(true) {
        polars_bail!(
            InvalidOperation: "cannot use values of type `{}` to replace values in a column of type `{}`",
            old.dtype(),
            s.dtype()
        )
    }
    execute_replacement(
        s,
        old,
        new,
        return_dtype.as_ref(),
        CastOptions::Strict,
        None,
    )
}

fn unpack_replacements(
    op: &str,
    s: &Series,
    old: &ListChunked,
    new: &ListChunked,
    default: Option<&Series>,
) -> PolarsResult<(Series, Series)> {
    let lengths = [("self", s.len()), ("old", old.len()), ("new", new.len())];
    find_output_length(
        op,
        lengths
            .into_iter()
            .chain(default.map(|value| ("default", value.len()))),
    )?;
    polars_ensure!(
        old.len() == 1 && new.len() == 1,
        nyi = "`{}` with a replacement pattern per row",
        op
    );
    let options = ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    };
    Ok((old.explode(options)?, new.explode(options)?))
}

/// Validate a deferred cast without registering categories in its output mapping.
fn deferred_cast_dtype(
    values: &Series,
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<DataType> {
    if dtype.is_categorical() && values.dtype().is_string() {
        return Ok(dtype.clone());
    }
    #[cfg_attr(not(feature = "dtype-categorical"), allow(unused_mut))]
    let mut swapped = Vec::<(DataType, DataType)>::new();
    let validation_dtype = dtype
        .clone()
        .map_leaves_with(values.dtype(), &mut |_source, dt| {
            #[cfg(feature = "dtype-categorical")]
            if let DataType::Categorical(cats, _) = &dt
                && _source != &dt
            {
                let own = DataType::from_categories(Categories::random(
                    cats.namespace().clone(),
                    CategoricalPhysical::U32,
                ));
                swapped.push((own.clone(), dt));
                return own;
            }
            dt
        });
    let validated = values.cast_with_options(&validation_dtype, options)?;
    Ok(validated.dtype().clone().map_leaves(&mut |dt| {
        swapped
            .iter()
            .find(|(own, _)| own == &dt)
            .map_or(dt, |(_, original)| original.clone())
    }))
}

/// Replaces the values of `s` in `old` by the paired `new` values, cast to `dtype`
/// (`new`'s own if `None`), and the others by `default`, raising if there is none.
fn execute_replacement(
    s: &Series,
    old: Series,
    new: Series,
    dtype: Option<&DataType>,
    cast_options: CastOptions,
    default: Option<&Series>,
) -> PolarsResult<Series> {
    polars_ensure!(
        new.len() == 1 || new.len() == old.len(),
        InvalidOperation: "`new` input for `replace` must have the same length as `old` or have length 1"
    );
    let (cast_old, kept) = cast_to_match(&old, s.dtype())?;
    // Checked once cast, as a cast can make values equal (e.g. "1" and "1.0" as floats). If values
    // were dropped, the cast can't have made those kept equal, so the values given are checked.
    validate_old(if kept.is_some() { &old } else { &cast_old })?;
    let old = cast_old;
    let dtype = dtype.unwrap_or(new.dtype()).clone();
    let dtype = &dtype;
    // A cast to a Categorical adds the values to its categories, so then only those used are cast.
    let defer = |values: &Series| dtype.contains_categoricals() && values.dtype() != dtype;
    let defer_cast = defer(&new);
    let (mut new, resolved) = if defer_cast {
        let resolved = deferred_cast_dtype(&new, dtype, cast_options)?;
        (new, resolved)
    } else {
        let new = new.cast_with_options(dtype, cast_options)?;
        let resolved = new.dtype().clone();
        (new, resolved)
    };
    // `default` is cast to `dtype` as `new` is, and the output dtype is the supertype of the two.
    let defer_default = default.is_some_and(defer);
    let default = match default {
        Some(d) if !defer_default && d.dtype() != dtype => {
            Some(d.cast_with_options(dtype, cast_options)?)
        },
        d => d.cloned(),
    };
    let out_dtype = match &default {
        Some(d) if !defer_default => try_get_supertype(&resolved, d.dtype())?,
        Some(d) if dtype.is_nested() || d.dtype().is_nested() => {
            // An unused incompatible default is allowed, but a successful cast
            // contributes its resolved type even when no row selects it.
            let default_dtype =
                deferred_cast_dtype(d, dtype, cast_options).unwrap_or_else(|_| resolved.clone());
            try_get_supertype(&resolved, &default_dtype)?
        },
        _ => resolved.clone(),
    };
    if let Some(kept) = kept.filter(|_| new.len() != 1) {
        new = new.filter(&kept)?;
    }
    let cast_used = |new: Series, used: &[bool]| match defer_cast {
        false => Ok(new),
        true if used.iter().all(|&used| used) => new.cast_with_options(&resolved, cast_options),
        true => {
            let filler = match used.iter().position(|&used| used) {
                Some(i) if !new.dtype().is_nested() => new.new_from_index(i, 1),
                _ => Series::full_null(PlSmallStr::EMPTY, 1, new.dtype()),
            };
            let used = BooleanChunked::from_slice(PlSmallStr::EMPTY, used);
            new.zip_with(&used, &filler)?
                .cast_with_options(&resolved, cast_options)
        },
    };
    // `default` for the values not `matched`, in the output dtype (if deferred, only those cast).
    let cast_default = |default: &Series, matched: &BooleanChunked| {
        if default.dtype() == &out_dtype {
            Ok(default.clone())
        } else if !defer_default {
            default.cast(&out_dtype)
        } else if matched.num_trues() == matched.len() {
            Ok(Series::full_null(PlSmallStr::EMPTY, 1, &out_dtype))
        } else {
            let selected = if default.len() == 1 {
                default.clone()
            } else {
                let nulls = Series::full_null(PlSmallStr::EMPTY, 1, default.dtype());
                nulls.zip_with_same_type(matched, default)?
            };
            // Ignored rows may have prevented validation; resolve the selected
            // values before combining their actual type with the inferred type.
            let selected = selected.cast_with_options(dtype, cast_options)?;
            let dtype = try_get_supertype(&out_dtype, selected.dtype())?;
            if selected.dtype() == &dtype {
                Ok(selected)
            } else {
                selected.cast(&dtype)
            }
        }
    };

    if old.is_empty() {
        // No `old` value has a category, so none can match.
        let Some(default) = default else {
            ensure_all_replaced(s, &old, 0)?;
            return Ok(Series::full_null(s.name().clone(), s.len(), &out_dtype));
        };
        let matched = BooleanChunked::full(PlSmallStr::EMPTY, false, s.len());
        return cast_default(&default, &matched)?.broadcast_owned_to(s.len());
    }
    if new.len() == 1 {
        let mask = get_replacement_mask(s, &old)?;
        let default = match &default {
            Some(default) => cast_default(default, &mask)?,
            None => {
                ensure_all_replaced(s, &old, mask.num_trues())?;
                Series::full_null(PlSmallStr::EMPTY, 1, &out_dtype)
            },
        };
        return cast_used(new, &[mask.any()])?.zip_with(&mask, &default);
    }

    // The index of the `old` value each value of `s` equals, if any.
    polars_ensure!(
        old.dtype() == s.dtype(),
        SchemaMismatch: "cannot replace values of type `{}` using keys of type `{}`",
        s.dtype(), old.dtype()
    );
    let (_, idx) = sort_or_hash_left(s, &old.rechunk(), false, JoinValidation::default(), true)?;
    #[cfg(feature = "chunked_ids")]
    let idx = idx.unwrap_left();
    let mut used = vec![false; new.len()];
    let mut unused = if defer_cast { new.len() } else { 0 };
    for i in idx.iter().filter(|i| !i.is_null_idx()) {
        if unused == 0 {
            break;
        }
        unused -= !std::mem::replace(&mut used[i.idx() as usize], true) as usize;
    }
    IdxCa::with_nullable_idx(&idx, |idx| {
        let missing = idx.null_count();
        if default.is_none() {
            ensure_all_replaced(s, &old, s.len() - missing)?;
        }
        let out = cast_used(new, &used)?.take(idx)?;
        match &default {
            Some(default) if missing > 0 => {
                let matched = idx.is_not_null();
                out.zip_with(&matched, &cast_default(default, &matched)?)
            },
            Some(_) if out.dtype() != &out_dtype => out.cast(&out_dtype),
            _ => Ok(out),
        }
    })
}

/// Validate the `old` input.
fn validate_old(old: &Series) -> PolarsResult<()> {
    polars_ensure!(
        old.n_unique()? == old.len(),
        InvalidOperation: "`old` input for `replace` must not contain duplicates"
    );
    Ok(())
}

/// Get a boolean mask of which values in the original Series will be replaced.
///
/// Null values are propagated to the mask, unless `old` contains a null.
fn get_replacement_mask(s: &Series, old: &Series) -> PolarsResult<BooleanChunked> {
    if old.null_count() == old.len() {
        // Fast path for when users are using `replace(None, ...)` instead of `fill_null`.
        Ok(s.is_null())
    } else if old.len() == 1 {
        Ok(s.equal(old)?)
    } else {
        let nulls_equal = old.null_count() > 0;
        let old = old.implode()?;
        is_in(s, &old.into_series(), nulls_equal)
    }
}

/// Casts `values` to `dtype` to match them against values of that dtype.
///
/// A value the cast rounds (2.5 to an integer, a datetime to its date) has no equal of that dtype,
/// so can't match, and is dropped. Casting strings (or the values of another Categorical or Enum)
/// to a Categorical would add them to its categories, so they are looked up instead: a string
/// without a category can't match, and is dropped too. If any value was dropped, the mask of the
/// kept values is also returned.
fn cast_to_match(
    values: &Series,
    dtype: &DataType,
) -> PolarsResult<(Series, Option<BooleanChunked>)> {
    #[cfg(feature = "dtype-categorical")]
    if let DataType::Categorical(_, mapping) = dtype
        && values.dtype() != dtype
        && matches!(
            values.dtype(),
            DataType::String | DataType::Categorical(..) | DataType::Enum(..)
        )
    {
        let strings = values.cast(&DataType::String)?;
        let kept: BooleanChunked = strings
            .str()?
            .iter()
            .map(|opt_s| opt_s.is_none_or(|s| mapping.get_cat(s).is_some()))
            .collect_ca_trusted(PlSmallStr::EMPTY);
        if !kept.all() {
            // Every string left has a category, so the cast adds none.
            return Ok((strings.filter(&kept)?.strict_cast(dtype)?, Some(kept)));
        }
    }
    let cast = values.strict_cast(dtype)?;
    // Not to a float, as the closest float is the one meant (`0.1` is the `Float32` 0.1), and not
    // a datetime to a time (its time of day), which can't be cast back.
    use DataType as D;
    let can_round = match (values.dtype(), dtype) {
        (from, to) if from == to => false,
        (from, to) if from.is_float() => to.is_integer() || to.is_decimal() || to.is_temporal(),
        (from, to) if from.is_decimal() => to.is_integer() || to.is_decimal(),
        (D::Datetime(..), D::Date | D::Datetime(..)) | (D::Duration(_), D::Duration(_)) => true,
        _ => false,
    };
    if can_round {
        let kept = cast.cast(values.dtype())?.equal_missing(values)?;
        if !kept.all() {
            return Ok((cast.filter(&kept)?, Some(kept)));
        }
    }
    Ok((cast, None))
}

/// Ensure that all values were replaced, given that `matched` were: nulls need not be unless `old`
/// has a null.
fn ensure_all_replaced(s: &Series, old: &Series, matched: usize) -> PolarsResult<()> {
    let unmatched_nulls = if old.null_count() > 0 {
        0
    } else {
        s.null_count()
    };
    polars_ensure!(
        matched + unmatched_nulls == s.len(),
        InvalidOperation: "incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."
    );
    Ok(())
}
