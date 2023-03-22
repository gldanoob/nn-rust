use crate::linalg::Matrix;

pub struct NN {
    layers: Vec<usize>,
    activations: Vec<Matrix>,
    z: Vec<Matrix>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
}

impl NN {
    pub fn new(layers: &[usize]) -> NN {
        let mut activations = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut z = Vec::new();

        // Input layer has no weights or biases
        weights.push(Matrix::new(0, 0));
        z.push(Matrix::new(0, 0));

        for dim in layers.windows(2) {
            weights.push(Matrix::new(dim[1], dim[0]).randomize());
            z.push(Matrix::new(dim[1], dim[0]));
        }
        for l in layers {
            activations.push(Matrix::new(*l, 1));
            biases.push(Matrix::new(*l, 1).randomize());
        }
        NN {
            layers: layers.to_vec(),
            activations,
            z,
            weights,
            biases,
        }
    }

    fn ff_to(&mut self, layer: usize) {
        self.z[layer] = self.weights[layer]
            .dot(&self.activations[layer - 1])
            .add(&self.biases[layer].clone());
        self.activations[layer] = self.z[layer].map(|x| Self::sigmoid(x));
    }

    fn calc_grad(self, dcda: &Matrix, layer: usize) -> (Matrix, Matrix, Matrix) {
        let dcdz = dcda.hadamard(&self.z[layer].map(|x| Self::dsigmoid(x)));
        let dcdb = dcdz.clone();
        let dcdw = dcdz.dot(&self.activations[layer - 1].transpose());
        let dcda = self.weights[layer].transpose().dot(&dcdz);

        return (dcdb, dcdw, dcda);
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn dsigmoid(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }

    pub fn run(&mut self, input: &[f64]) {
        assert!(input.len() == self.layers[0]);
        self.activations[0] = Matrix::from(self.layers[0], 1, input);
        for i in 1..self.layers.len() {
            self.ff_to(i);
        }
    }

    pub fn output(self) -> Matrix {
        self.activations[self.layers.len() - 1].clone()
    }
}
