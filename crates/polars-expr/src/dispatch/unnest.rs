use std::ops::Range;

use polars_arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use polars_arrow::array::{Array, FixedSizeListArray, ListArray};
use polars_arrow::bitmap::bitmask::BitMask;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::OffsetsBuffer;
use polars_core::prelude::*;
use polars_plan::plans::unnest_dtype;

/// Flatten `depth` inner List/Array levels into the rows.
///
/// A null inner list becomes one null element and a null inner array becomes `width` null
/// elements, so unnesting by two equals unnesting by one twice. Null rows stay null.
pub(super) fn unnest(c: &Column, depth: usize) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| unnest_series(s, depth))
}

fn unnest_series(s: &Series, depth: usize) -> PolarsResult<Series> {
    let dtype = unnest_dtype(s.dtype(), depth)?;
    if depth == 0 {
        return Ok(s.clone());
    }
    let arrow_dtype = dtype.to_physical().to_arrow(CompatLevel::newest());
    let chunks = s
        .chunks()
        .iter()
        .map(|arr| unnest_array(arr.as_ref(), depth, &arrow_dtype))
        .collect();
    // SAFETY: the chunks are built with the physical type of `dtype`.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(s.name().clone(), chunks, &dtype) })
}

#[derive(Clone, Copy)]
enum Layout<'a> {
    List(&'a [i64]),
    Array(usize),
}

impl Layout<'_> {
    /// Index of the first child of element `i`.
    #[inline(always)]
    fn child(self, i: usize) -> usize {
        match self {
            Self::List(offsets) => offsets[i] as usize,
            Self::Array(width) => i * width,
        }
    }
}

/// A List/Array level, restricted to the elements `start..end` reachable from the rows.
struct Level<'a> {
    layout: Layout<'a>,
    /// Only set if a reachable element is null.
    validity: Option<&'a Bitmap>,
    /// The validity as a mask, for fast scans of its runs.
    mask: Option<BitMask<'a>>,
    start: usize,
    end: usize,
    /// Number of leaf nulls a null element expands to: its own and deeper Array widths.
    null_width: usize,
}

impl Level<'_> {
    /// Runs of valid or null elements of `start..end` (at most 32 long if there are nulls).
    #[inline(always)]
    fn runs(&self, start: usize, end: usize) -> impl Iterator<Item = (bool, Range<usize>)> {
        let mask = self.mask;
        let mut i = start;
        std::iter::from_fn(move || {
            (i < end).then(|| {
                let Some(mask) = mask else {
                    i = end;
                    return (true, start..end);
                };
                let bits = mask.get_u32(i);
                let is_valid = bits & 1 == 1;
                let len = if is_valid {
                    bits.trailing_ones()
                } else {
                    bits.trailing_zeros()
                };
                let run = i..end.min(i + len as usize);
                i = run.end;
                (is_valid, run)
            })
        })
    }
}

fn unnest_array(outer: &dyn Array, depth: usize, dtype: &ArrowDataType) -> ArrayRef {
    let mut levels = Vec::with_capacity(depth + 1);
    let (mut arr, mut start, mut end) = (outer, 0, outer.len());
    for _ in 0..=depth {
        let (layout, values) = match arr.as_any().downcast_ref::<ListArray<i64>>() {
            Some(list) => (Layout::List(list.offsets().as_slice()), list.values()),
            None => {
                let arr = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                (Layout::Array(arr.size()), arr.values())
            },
        };
        let validity = arr
            .validity()
            .filter(|v| v.null_count_range(start, end - start) > 0);
        levels.push(Level {
            layout,
            validity,
            mask: validity.map(BitMask::from_bitmap),
            start,
            end,
            null_width: 1,
        });
        (arr, start, end) = (values.as_ref(), layout.child(start), layout.child(end));
    }
    let (leaf, leaf_range) = (arr, start..end);
    let mut null_width = 1usize;
    for level in levels.iter_mut().rev() {
        if let Layout::Array(width) = level.layout {
            null_width = null_width.saturating_mul(width);
        }
        level.null_width = null_width;
    }

    let rows = outer.len();
    let row_validity = outer.validity().cloned();
    let inner = &levels[1..];
    // Inner nulls only need a copy if a List level lies below them; otherwise each null element
    // covers a fixed, contiguous run of leaf slots that its validity can mask in place.
    let copy_free = inner
        .iter()
        .rev()
        .skip_while(|level| matches!(level.layout, Layout::Array(_)))
        .all(|level| level.validity.is_none());
    if !copy_free {
        let (values, offsets) = flatten_with_nulls(&levels, leaf, leaf_range);
        // SAFETY: the offsets are the running output length per row.
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
        return ListArray::<i64>::new(dtype.clone(), offsets, values, row_validity).boxed();
    }

    let mut values = leaf.sliced(leaf_range.start, leaf_range.len());
    if inner.iter().any(|level| level.validity.is_some()) {
        let mut validity = match values.validity() {
            Some(validity) => validity.clone().make_mut(),
            None => MutableBitmap::from_len_set(values.len()),
        };
        for level in inner {
            let Some(level_validity) = level.validity else {
                continue;
            };
            // Each null element masks the `null_width` leaf slots it spans.
            let len = level.end - level.start;
            let nulls = !&level_validity.clone().sliced(level.start, len);
            for i in nulls.true_idx_iter() {
                for slot in i * level.null_width..(i + 1) * level.null_width {
                    // SAFETY: the slots of the reachable elements make up the leaf range.
                    unsafe { validity.set_unchecked(slot, false) };
                }
            }
        }
        values = values.with_validity(Some(validity.freeze()));
    }
    match dtype {
        ArrowDataType::FixedSizeList(..) => {
            FixedSizeListArray::new(dtype.clone(), rows, values, row_validity).boxed()
        },
        _ => {
            let offsets = compose_offsets(&levels, leaf_range.start);
            // SAFETY: composing monotonic offsets keeps them monotonic.
            let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
            ListArray::<i64>::new(dtype.clone(), offsets, values, row_validity).boxed()
        },
    }
}

/// Leaf offsets of every row, rebased to start at zero.
fn compose_offsets(levels: &[Level], leaf_start: usize) -> Vec<i64> {
    let mut offsets: Vec<i64> = match levels[0].layout {
        Layout::List(offsets) => offsets.to_vec(),
        Layout::Array(width) => (0..=levels[0].end).map(|i| (i * width) as i64).collect(),
    };
    for level in &levels[1..] {
        match level.layout {
            Layout::List(child) => offsets.iter_mut().for_each(|o| *o = child[*o as usize]),
            Layout::Array(width) => offsets.iter_mut().for_each(|o| *o *= width as i64),
        }
    }
    offsets.iter_mut().for_each(|o| *o -= leaf_start as i64);
    offsets
}

/// Copy the leaf values of each valid row, expanding null inner elements to their
/// `null_width` nulls and skipping any values hidden under nulls.
fn flatten_with_nulls(
    levels: &[Level],
    leaf: &dyn Array,
    leaf_range: Range<usize>,
) -> (ArrayRef, Vec<i64>) {
    let (rows, inner) = levels.split_first().unwrap();
    let (first, rest) = inner.split_first().unwrap();
    let mut sink = LeafSink {
        leaf,
        builder: make_builder(leaf.dtype()),
        run: 0..0,
        len: 0,
    };
    sink.builder.reserve(leaf_range.len());
    let mut offsets = Vec::with_capacity(rows.end + 1);
    offsets.push(0);
    for (is_valid, run) in rows.runs(0, rows.end) {
        for row in run {
            if is_valid {
                let (start, end) = (rows.layout.child(row), rows.layout.child(row + 1));
                sink.flatten_level(first, rest, start, end);
            }
            offsets.push(sink.len as i64);
        }
    }
    sink.flush();
    (sink.builder.freeze(), offsets)
}

/// Appends leaf ranges and nulls to a builder, copying adjacent ranges in one go.
struct LeafSink<'a> {
    leaf: &'a dyn Array,
    builder: Box<dyn ArrayBuilder>,
    /// Pending leaf range, not yet copied.
    run: Range<usize>,
    len: usize,
}

impl LeafSink<'_> {
    /// Append the leaves of elements `start..end` of the first of `levels`.
    fn flatten(&mut self, levels: &[Level], start: usize, end: usize) {
        match levels.split_first() {
            Some((level, rest)) => self.flatten_level(level, rest, start, end),
            None => self.extend(start, end),
        }
    }

    #[inline(always)]
    fn flatten_level(&mut self, level: &Level, rest: &[Level], start: usize, end: usize) {
        for (is_valid, run) in level.runs(start, end) {
            if is_valid {
                self.descend(level, rest, run.start, run.end);
            } else {
                self.flush();
                self.builder.extend_nulls(run.len() * level.null_width);
                self.len += run.len() * level.null_width;
            }
        }
    }

    /// Append the leaves of the valid elements `start..end`, whose children are contiguous.
    #[inline(always)]
    fn descend(&mut self, level: &Level, rest: &[Level], start: usize, end: usize) {
        let (first, last) = (level.layout.child(start), level.layout.child(end));
        if rest.is_empty() {
            self.extend(first, last);
        } else {
            self.flatten(rest, first, last);
        }
    }

    #[inline(always)]
    fn extend(&mut self, start: usize, end: usize) {
        if start != self.run.end {
            self.flush();
            self.run.start = start;
        }
        self.run.end = end;
        self.len += end - start;
    }

    fn flush(&mut self) {
        if !self.run.is_empty() {
            self.builder.subslice_extend(
                self.leaf,
                self.run.start,
                self.run.len(),
                ShareStrategy::Always,
            );
        }
        self.run.start = self.run.end;
    }
}
