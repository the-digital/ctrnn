pub mod activation;

use activation::sigmoid;

/// Continuous-Time Recurrent Neural Network (`CTRNN`) implementation in Rust.
///
/// # Example usage
/// ```rust
/// use ctrnn::CTRNN;
///
/// let mut ctrnn = CTRNN::new(6);
/// ctrnn.tick();
/// ```
pub struct CTRNN {
    /// Number of nodes in the network
    count: usize,
    /// Time constant used for conversion from continuous to discrete time
    time_constant: f64,
    /// Array of each node's activation
    nodes: Vec<f64>,
    /// Array of each node's ***input*** weights
    /// e.g. `weights[i][j]` is the weight going from node `j` TO node `i`
    weights: Vec<Vec<f64>>,
    /// Array of each node's bias
    /// This can be thought of as how stimulated a neuron must be
    biases: Vec<f64>,
}

impl CTRNN {
    /// Create a new CTRNN instance with the specified amount of nodes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ctrnn::CTRNN;
    ///
    /// let mut ctrnn = CTRNN::new(6);
    /// ```
    /// > This will create a fully-connected CTRNN with 6 nodes.
    pub fn new(nodes: usize) -> Self {
        Self {
            count: nodes,
            time_constant: 0.05,
            nodes: vec![0.0; nodes],
            weights: vec![vec![0.0; nodes]; nodes],
            biases: vec![0.0; nodes],
        }
    }

    /// Adjust the time constant `tau` for the network.
    /// > Used when converting from continuous time to discrete time.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::CTRNN::new(6);
    /// ctrnn.set_time_constant(0.05);
    /// ```
    /// > This will adjust the `time constant` to a value of `0.05`.
    pub fn set_time_constant(&mut self, dt: f64) -> &mut Self {
        self.time_constant = dt;
        self
    }

    /// Execute a single calculation cycle.
    /// > Temporarily the complexity is `O(n^2)`
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::CTRNN::new(6);
    /// ctrnn.tick();
    /// ```
    pub fn tick(&mut self) {
        let dt = self.time_constant;
        let nodes = self.nodes.clone();
        self.nodes.clear();
        for i in 0 .. self.count {
            let weights = self.weights.get(i).unwrap();
            let mut sum: f64 = 0.0;
            for j in 0 .. self.count {
                let weight = weights.get(j).unwrap();
                if weight == &0.0 { continue; }
                let bias = self.biases.get(j).unwrap();
                let presynaptic = nodes.get(j).unwrap();

                sum += weight * sigmoid(presynaptic - bias);
            }

            let postsynaptic = nodes.get(i).unwrap();
            self.nodes.push(postsynaptic + (sum - postsynaptic) / dt);
        }
    }
}

#[cfg(test)]
mod ctrnn {
    use super::CTRNN;

    #[test]
    fn creation() {
        let ctrnn = CTRNN::new(6);
        assert_eq!(ctrnn.count, 6);
        assert_eq!(ctrnn.nodes.len(), 6);
        assert_eq!(ctrnn.weights.len(), 6);
        assert_eq!(ctrnn.biases.len(), 6);
    }
}
