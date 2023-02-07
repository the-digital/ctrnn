use std::{ops::Range, f64::consts::PI};

#[derive(Clone)]
pub struct Fluctuator {
    pub center: f64,
    pub range: Range<f64>,
    pub range_period: Range<f64>,
    pub range_amplitude: Range<f64>,

    pub period: f64,
    pub amplitude: f64,
    pub time: f64,

    pub convergence_rate: f64,
    pub learning_rate: f64,
}

impl Fluctuator {
    pub fn new(center: f64) -> Self {
        let mut flux = Self {
            center,
            ..Default::default()
        };
        flux.randomize_period();
        flux
    }

    pub fn get(&self) -> f64 {
        let theta = self.time * 2.0 * PI / self.period;
        self.center + self.amplitude * theta.sin()
    }

    fn randomize_period(&mut self) {
        let diff = self.range_period.end - self.range_period.start;
        let p = self.range_period.start + diff * 0.0;
        self.period = (p * 10.0).floor() / 10.0;
        self.time = 0.0;
    }

    pub fn update(&mut self, dt: f64, reward: f64) -> f64 {
        self.amplitude -= self.convergence_rate * self.range_amplitude.end * reward;
        self.amplitude = self.range_amplitude.clamp(self.amplitude);

        let theta = self.time * 2.0 * PI / self.period;
        let d = self.amplitude * theta.sin();
        self.center += self.learning_rate * d * reward;

        self.time += dt;
        if self.time > self.period { self.randomize_period(); }
        d
    }
}

impl Default for Fluctuator {
    fn default() -> Self {
        Self {
            center: 0.0,
            range: -16.0..16.0,
            range_period: 3.0..12.0,
            range_amplitude: 0.001..10.0,
            period: 0.0,
            amplitude: 0.0,
            time: 0.0,
            convergence_rate: 0.1,
            learning_rate: 0.1,
        }
    }
}

impl Into<f64> for Fluctuator {
    fn into(self) -> f64 { self.get() }
}

impl From<f64> for Fluctuator {
    fn from(value: f64) -> Self { Self::new(value) }
}

trait Clamp<T> {
    fn clamp(&self, value: T) -> T;
}

impl Clamp<f64> for Range<f64> {
    fn clamp(&self, value: f64) -> f64 {
        self.start.max(self.end.min(value))
    }
}
