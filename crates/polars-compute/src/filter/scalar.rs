use arrow::bitmap::Bitmap;
use bytemuck::Pod;
use polars_utils::slice::load_padded_le_u64;

/// # Safety
/// If the ith bit of m is set (m & (1 << i)), then v[i] must be in-bounds.
/// out must be valid for at least m.count_ones() + 1 writes.
unsafe fn scalar_sparse_filter_unchecked<T: Pod>(v: &[T], mut m: u64, out: *mut T) {
    let mut written = 0usize;

    while m > 0 {
        // Unroll loop manually twice.
        let idx = m.trailing_zeros() as usize;
        *out.add(written) = *v.get_unchecked(idx);
        m &= m.wrapping_sub(1); // Clear least significant bit.
        written += 1;

        // tz % 64 otherwise we could go out of bounds
        let idx = (m.trailing_zeros() % 64) as usize;
        *out.add(written) = *v.get_unchecked(idx);
        m &= m.wrapping_sub(1); // Clear least significant bit.
        written += 1;
    }
}

/// # Safety
/// v.len() >= 64 must hold.
/// out must be valid for at least m.count_ones() + 1 writes.
unsafe fn scalar_dense_filter_unchecked<T: Pod>(v: &[T], mut m: u64, out: *mut T) {
    // Rust generated significantly better code if we write the below loop
    // with v as a pointer, and out.add(written) instead of incrementing out
    // directly.
    let mut written = 0usize;
    let mut src = v.as_ptr();

    // We hope the outer loop doesn't get unrolled, but the inner loop does.
    for _ in 0..16 {
        for i in 0..4 {
            *out.add(written) = *src;
            written += ((m >> i) & 1) as usize;
            src = src.add(1);
        }
        m >>= 4;
    }
}

/// # Safety
/// out must be valid for at least mask.count_ones() + 1 writes.
pub unsafe fn filter_scalar_values<T: Pod>(values: &[T], mask: &Bitmap, mut out: *mut T) {
    assert_eq!(values.len(), mask.len());
    let (mut mask_bytes, offset, len) = mask.as_slice();

    // Handle offset.
    let mut value_idx = 0;
    if offset > 0 {
        let first_byte = mask_bytes[0];
        mask_bytes = &mask_bytes[1..];

        for byte_idx in offset..8 {
            let mask_bit = first_byte & (1 << byte_idx) != 0;
            if mask_bit && value_idx < len {
                unsafe {
                    // SAFETY: we only write to out for one bits, and checked
                    // that value_idx < len.
                    *out = *values.get_unchecked(value_idx);
                    out = out.add(1);
                }
            }
            value_idx += 1;
        }
    }

    // Handle bulk.
    while value_idx + 64 <= len {
        let (mask_chunk, value_chunk);
        unsafe {
            // SAFETY: we checked that value_idx + 64 <= len, so these are
            // all in-bounds.
            mask_chunk = mask_bytes.get_unchecked(0..8);
            mask_bytes = mask_bytes.get_unchecked(8..);
            value_chunk = values.get_unchecked(value_idx..value_idx + 64);
            value_idx += 64;
        };
        let m = u64::from_le_bytes(mask_chunk.try_into().unwrap());

        // Fast-path: empty mask.
        if m == 0 {
            continue;
        }

        unsafe {
            // SAFETY: we will only write at most m_popcnt + 1 to out, which
            // is allowed.

            // Fast-path: completely full mask.
            if m == u64::MAX {
                core::ptr::copy_nonoverlapping(value_chunk.as_ptr(), out, 64);
                out = out.add(64);
                continue;
            }

            let m_popcnt = m.count_ones();
            if m_popcnt <= 16 {
                scalar_sparse_filter_unchecked(value_chunk, m, out)
            } else {
                scalar_dense_filter_unchecked(value_chunk, m, out)
            };
            out = out.add(m_popcnt as usize);
        }
    }

    if value_idx < len {
        let rest_len = len - value_idx;
        assert!(rest_len < 64);
        let m = load_padded_le_u64(mask_bytes) & ((1 << rest_len) - 1);
        unsafe {
            let value_chunk = values.get_unchecked(value_idx..);
            scalar_sparse_filter_unchecked(value_chunk, m, out);
        }
    }
}
