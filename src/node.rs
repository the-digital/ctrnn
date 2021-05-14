#[derive(Clone)]
pub struct Node {
    pub bias: f64,
    pub time_constant: f64,
}

impl Node {
    pub fn new(bias: f64, dt: f64) -> Self {
        Self { bias, time_constant: dt }
    }
}

impl Default for Node { fn default() -> Self { Self::new(0.0, 1.0) } }
