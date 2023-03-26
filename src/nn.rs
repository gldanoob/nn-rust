use crate::linalg::Matrix;

pub struct NN {
    layers: Vec<usize>,
    a: Vec<Matrix>,
    z: Vec<Matrix>,
    w: Vec<Matrix>,
    b: Vec<Matrix>,

    // Store the gradient of the cost function with respect to the weights and biases of each layer
    dcdb: Vec<Matrix>,
    dcdw: Vec<Matrix>,
}

impl NN {
    // layers: [2, 5, 1] means 2 input neurons, 5 hidden neurons, 1 output neuron
    pub fn new(layers: &[usize]) -> NN {
        if layers.len() < 2 {
            panic!("At least 2 layers required (Input and output layer)");
        }

        // a = activation (array of column vectors) (input layer excluded)
        // w = weights (array of matrices)
        // b = biases (array of column vectors)
        // z = weighted sum (array of column vectors)
        let mut a = Vec::new();
        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut z = Vec::new();

        // Sum of gradients over a batch of training samples
        let mut dcdw = Vec::new();
        let mut dcdb = Vec::new();

        for dim in layers.windows(2) {
            w.push(Matrix::new(dim[1], dim[0]).randomize());
            dcdw.push(Matrix::new(dim[1], dim[0]));
        }
        for l in layers[1..].iter() {
            z.push(Matrix::new(*l, 1));
            a.push(Matrix::new(*l, 1));
            b.push(Matrix::new(*l, 1).randomize());
            dcdb.push(Matrix::new(*l, 1));
        }
        NN {
            layers: layers.to_vec(),
            a,
            z,
            w,
            b,
            dcdb,
            dcdw,
        }
    }

    // feed forward to layer given input
    pub fn feedforward(&mut self, input: &Matrix) {
        for i in self.layer_indices() {
            // activation of previous layer
            let a_prev = if i == 0 { input } else { &self.a[i - 1] };

            self.z[i] = self.w[i].dot(a_prev).add(&self.b[i].clone());
            self.a[i] = self.z[i].map(|x| Self::sigmoid(x));
        }
    }

    // Calculate the gradient of the cost function with respect to the weights and biases of all layers
    fn backpropagate(&mut self, target: &Matrix) {
        // For output layer, dc/da = 2(a - y) where y is the target
        // But I ignored the coefficient which will be cancelled out when calculating the gradient descent
        let dcda = self.output().sub(&target);

        for i in self.layer_indices().rev() {
            let dcdz = dcda.hadamard(&self.z[i].map(|x| Self::dsigmoid(x)));
            let dcdb = dcdz.clone();
            let dcdw = dcdz.dot(&self.a[i - 1].transpose());
            let dcda = self.w[i].transpose().dot(&dcdz);

            // Update total gradients
            self.dcdb[i] = self.dcdb[i].add(&dcdb);
            self.dcdw[i] = self.dcdw[i].add(&dcdw);
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // Derivative of sigmoid function
    fn dsigmoid(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }

    pub fn run(&mut self, inputs: &Matrix) -> Matrix {
        if inputs.rows() != self.layers[0] {
            panic!("Number of inputs must be the same as the number of input neurons");
        }

        self.feedforward(inputs);
        self.output()
    }

    // Train the network with a batch of training samples
    pub fn train_batch(&mut self, inputs: &Matrix, targets: &Matrix, lr: f64) {
        // Check dimensions
        if inputs.rows() != targets.rows() {
            panic!("Number of inputs and targets must be the same");
        }

        if inputs.cols() != self.layers[0] {
            panic!("Number of inputs must be the same as the number of input neurons");
        }

        for i in 0..inputs.rows() {
            let input = inputs.get_row(i);
            let target = targets.get_row(i);

            // Updates self.a and self.z
            self.feedforward(&input);

            // Updates self.dcdw and self.dcdb
            self.backpropagate(&target);
        }

        // Update weights and biases with the total gradients * learning rate
        for i in self.layer_indices() {
            self.b[i] = self.b[i].sub(&self.dcdb[i].scale(lr));
            self.w[i] = self.w[i].sub(&self.dcdw[i].scale(lr));
        }
    }

    // Indices of dense layers (excluding input layer)
    fn layer_indices(&self) -> std::ops::Range<usize> {
        0..self.layers.len() - 1
    }

    // Get the output of the network
    pub fn output(&self) -> Matrix {
        self.a.last().unwrap().clone()
    }
}
