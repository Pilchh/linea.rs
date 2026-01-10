#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
