pub type ActivationFunc = fn(f64) -> f64;

/// A logistic curve that tends towards `0` for negative inputs and `1` for positive inputs.
pub fn sigmoid(x: f64) -> f64 {
    (1.0 + (-x).exp()).recip()
}

pub fn inverse_sigmoid(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// A curve that is `0` if the input is negative, and `x` if it is positive.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}
