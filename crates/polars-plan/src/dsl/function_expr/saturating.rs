use super::*;

pub(super) fn saturating_add(s: &[Series]) -> PolarsResult<Series> {
    let lhs = s[0].clone();
    let rhs = s[1].clone();

    Ok(lhs.saturating_add(rhs))
}

pub(super) fn saturating_sub(s: &[Series]) -> PolarsResult<Series> {
    let lhs = s[0].clone();
    let rhs = s[1].clone();

    Ok(lhs.saturating_sub(rhs))
}
