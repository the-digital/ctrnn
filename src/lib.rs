pub mod node;
pub mod activation;

use activation::sigmoid;

/// Continuous-Time Recurrent Neural Network (`CTRNN`) implementation in Rust.
///
/// # Example usage
/// ```rust
/// use ctrnn::CTRNN;
///
/// let mut ctrnn = CTRNN::new(3);
/// ctrnn.tick(vec![1.0, 0.0, 0.0], 1.0 / 60.0);
/// ```
pub struct CTRNN {
    /// Number of nodes in the network
    count: usize,
    /// Array of each node's parameters:
    /// - `bias`: how stimulated a neuron must be before activating
    /// - `time_constant`: the excitatory component of a neuron
    nodes: Vec<node::Node>,
    /// Array of each node's ***input*** weights
    /// e.g. `weights[i][j]` is the weight going from node `j` TO node `i`
    weights: Vec<Vec<f64>>,
    /// Array of each node's activation
    states: Vec<f64>,
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
    /// > This will create a fully-connected CTRNN with 6 nodes
    pub fn new(nodes: usize) -> Self {
        Self {
            count: nodes,
            nodes: vec![node::Node::default(); nodes],
            weights: vec![vec![0.0; nodes]; nodes],
            states: vec![0.0; nodes],
        }
    }

    /// Set an individual weight within the network.
    /// > By default all weights are initialized to `0.0`
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::CTRNN::new(3);
    /// ctrnn.set_weight(0, 2, 1.0);
    /// ctrnn.set_weight(1, 2, 1.0);
    /// ```
    /// > This will make it so node `2` gets excited from nodes `0` and `1`
    pub fn set_weight(
        &mut self,
        from: usize,
        to: usize,
        weight: f64
    ) -> &mut Self {
        self.weights[to][from] = weight;
        self
    }

    /// Adjust the bias `theta` for a given neuron
    /// > Can be thought of as how stimulated a neuron must be to activate
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::CTRNN::new(6);
    /// ctrnn.set_bias(0, 1.1);
    /// ```
    /// > This will cause node `0` to have a slight dampening effect
    pub fn set_bias(&mut self, node: usize, bias: f64) -> &mut Self {
        self.nodes[node].bias = bias;
        self
    }

    /// Adjust the time constant `tau` for a given neuron
    /// > Can be thought of as the neuron's excitatory component
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::CTRNN::new(6);
    /// ctrnn.set_time_constant(0, 1.1);
    /// ```
    /// > This will cause node `0` to have a slight dampening effect
    pub fn set_time_constant(&mut self, node: usize, dt: f64) -> &mut Self {
        self.nodes[node].time_constant = dt;
        self
    }

    /// Execute a single calculation cycle.
    /// > Temporarily the complexity is `O(n^2)`
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut ctrnn = ctrnn::CTRNN::new(3);
    /// ctrnn.tick(vec![1.0, 0.0, 0.0], 1.0 / 60.0);
    /// ```
    pub fn tick(&mut self, inputs: Vec<f64>, dt: f64) {
        let states = self.states.clone();
        self.states.clear();
        for i in 0 .. self.count {
            let weights = &self.weights[i];
            let postsynaptic = states[i];
            let mut sum: f64 = 0.0;
            for j in 0 .. self.count {
                let weight = weights[j];
                if weight == 0.0 { continue; }
                let presynaptic = states[j];
                let bias = self.nodes[j].bias;

                sum += weight * sigmoid(presynaptic - bias);
            }

            let dy = (sum - postsynaptic + inputs[i]) / self.nodes[i].time_constant;
            self.states.push(postsynaptic + dy * dt);
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
        assert_eq!(ctrnn.states.len(), 6);
    }
}
