use crate::{fluctuator::Fluctuator, activation::sigmoid};

pub struct RLCTRNN {
    count: usize,
    biases: Vec<Fluctuator>,
    time_constants: Vec<Fluctuator>,
    weights: Vec<Vec<Fluctuator>>
}

impl RLCTRNN {
    /// Create a new RLCTRNN instance with the specified amount of nodes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ctrnn::RLCTRNN;
    ///
    /// let mut ctrnn = RLCTRNN::new(6);
    /// ```
    /// > This will create a fully-connected RLCTRNN with 6 nodes
    pub fn new(nodes: usize) -> Self {
        let mut ctrnn = Self {
            count: nodes,
            biases: vec![],
            time_constants: vec![],
            weights: vec![],
        };

        for _ in 0..nodes {
            ctrnn.biases.push(Fluctuator::from(0.0));
            ctrnn.time_constants.push(Fluctuator::from(1.0));
            let mut weights = vec![];
            for _ in 0..nodes { weights.push(Fluctuator::from(0.0)); }
            ctrnn.weights.push(weights);
        }

        ctrnn
    }

    /// Adjust the bias `theta` for a given neuron
    /// > Can be thought of as how stimulated a neuron must be to activate
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::RLCTRNN::new(6);
    /// ctrnn.set_bias(0, 1.1);
    /// ```
    /// > This will cause node `0` to have a slight dampening effect
    pub fn set_bias(&mut self, index: usize, value: f64) -> &mut Self {
        self.biases[index].center = value;
        self
    }

    /// Adjust the time constant `tau` for a given neuron
    /// > Can be thought of as the neuron's excitatory component
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::RLCTRNN::new(6);
    /// ctrnn.set_time_constant(0, 1.1);
    /// ```
    /// > This will cause node `0` to have a slight dampening effect
    pub fn set_time_constant(&mut self, index: usize, value: f64) -> &mut Self {
        self.time_constants[index].center = value;
        self
    }

    /// Set an individual weight within the network.
    /// > By default all weights are initialized to `0.0`
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::RLCTRNN::new(3);
    /// ctrnn.set_weight(0, 2, 1.0);
    /// ctrnn.set_weight(1, 2, 1.0);
    /// ```
    /// > This will make it so node `2` gets excited from nodes `0` and `1`
    pub fn set_weight(&mut self, from: usize, to: usize, value: f64) -> &mut Self {
        self.weights[to][from].center = value;
        self
    }

    pub fn add_node(&mut self) -> &mut Self {
        self
    }

    pub fn update(&mut self, dt: f64, voltages: Vec<f64>, inputs: Vec<f64>) -> Vec<f64> {
        (0..self.count)
            .map(|i| voltages[i] + self.get_delta(&voltages, i) * dt + inputs.get(i).unwrap_or(&0.0))
            .collect()
    }

    pub fn get_outputs(&self, voltages: &Vec<f64>) -> Vec<f64> {
        (0..self.count).map(|i| sigmoid(voltages[i] + self.biases[i].get())).collect()
    }

    pub fn init_voltage(&self) -> Vec<f64> {
        (0..self.count).map(|_| 0.0).collect()
    }

    fn get_delta(&self, voltages: &Vec<f64>, index: usize) -> f64 {
        let weights = &self.weights[index];
        let mut sum = 0.0;
        for j in 0..self.count {
            let activation = sigmoid(voltages[j] + self.biases[j].get());
            sum += weights[j].get() * activation
        }
        (sum - voltages[index]) / self.time_constants[index].get()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn creation() {
        let ctrnn = RLCTRNN::new(6);
        assert_eq!(ctrnn.count, 6);
        assert_eq!(ctrnn.biases.len(), 6);
        assert_eq!(ctrnn.time_constants.len(), 6);
        assert_eq!(ctrnn.weights.len(), 6);
    }
}
