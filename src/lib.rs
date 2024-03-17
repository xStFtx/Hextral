extern crate rand;
extern crate nalgebra;

use nalgebra::{DVector, DMatrix};
use rand::Rng;

pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU(f64),
    ELU(f64),
}

pub enum Regularization {
    L2(f64),
    L1(f64),
    Dropout(f64),
}

pub enum Optimizer {
    SGD,
    SGDMomentum(f64),
    RMSProp(f64, f64),
    Adam(f64, f64),
}

pub struct BatchNormalization {
    running_mean: DVector<f64>,
    running_var: DVector<f64>,
    eps: f64,
}

impl BatchNormalization {
    pub fn new(size: usize, eps: f64) -> Self {
        BatchNormalization {
            running_mean: DVector::zeros(size),
            running_var: DVector::repeat(size, 1.0),
            eps,
        }
    }

    pub fn forward(&mut self, input: &DVector<f64>, training: bool) -> DVector<f64> {
        if training {
            let mean = input.mean();
            let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() - 1) as f64;
            let sqrt_var = var.sqrt() + self.eps;
            let normalized = input.map(|x| (x - mean) / sqrt_var);
            self.running_mean = (&self.running_mean * 0.9) + (mean * 0.1); // Fixed
            self.running_var = (&self.running_var * 0.9) + (var * 0.1); // Fixed
            normalized.component_mul(&(self.running_var.map(|x| 1.0 / (x.sqrt() + self.eps)))) // Apply normalization
        } else {
            let normalized = input.map(|x| (x - &self.running_mean) / (&self.running_var.map(|x| x.sqrt()) + self.eps));
            normalized
        }
    }

    pub fn backward(&self, input: &DVector<f64>, grad_output: &DVector<f64>) -> DVector<f64> {
        let normalized = input.map(|x| (x - &self.running_mean) / (&self.running_var.map(|x| x.sqrt()) + self.eps));
        let grad_gamma = grad_output.component_mul(&normalized);
        let grad_beta = grad_output.clone();
        let grad_input = grad_output.component_mul(&(1.0 / (&self.running_var.map(|x| x.sqrt()) + self.eps)));
        grad_input
    }
}

pub struct Hextral {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
    batch_norms: Vec<BatchNormalization>,
    activation: ActivationFunction,
    optimizer: Optimizer,
    optimizer_state: Vec<(DMatrix<f64>, DVector<f64>)>,
}

impl Hextral {
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        activation: ActivationFunction,
        optimizer: Optimizer,
    ) -> Self {
        let mut layers = Vec::with_capacity(hidden_sizes.len() + 1);
        let mut batch_norms = Vec::with_capacity(hidden_sizes.len());
        let mut optimizer_state = Vec::with_capacity(hidden_sizes.len() + 1);

        let mut prev_size = input_size;
        for &size in hidden_sizes {
            layers.push((
                DMatrix::from_fn(size, prev_size, |_, _| rand::thread_rng().gen::<f64>() * 0.1),
                DVector::zeros(size),
            ));
            batch_norms.push(BatchNormalization::new(size, 1e-5));
            optimizer_state.push((DMatrix::zeros(size, prev_size), DVector::zeros(size)));
            prev_size = size;
        }
        layers.push((
            DMatrix::from_fn(output_size, prev_size, |_, _| rand::thread_rng().gen::<f64>() * 0.1),
            DVector::zeros(output_size),
        ));
        optimizer_state.push((DMatrix::zeros(output_size, prev_size), DVector::zeros(output_size)));

        Hextral {
            layers,
            batch_norms,
            activation,
            optimizer,
            optimizer_state,
        }
    }

    pub fn forward_pass(&self, input: &DVector<f64>, training: bool) -> DVector<f64> {
        let mut output = input.clone();
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            output = weight * &output + bias;
            if i < self.batch_norms.len() {
                output = self.batch_norms[i].forward(&output, training);
            }
            output = match self.activation {
                ActivationFunction::Sigmoid => output.map(|x| sigmoid(x)),
                ActivationFunction::ReLU => output.map(|x| x.max(0.0)),
                ActivationFunction::Tanh => output.map(|x| x.tanh()),
                ActivationFunction::LeakyReLU(alpha) => output.map(|x| if x >= 0.0 { x } else { alpha * x }),
                ActivationFunction::ELU(alpha) => output.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }),
            };
        }
        output
    }

    pub fn backward_pass(&mut self, input: &DVector<f64>, output: &DVector<f64>, target: &DVector<f64>) {
        let mut grad_output = 2.0 * (output - target);
        for i in (0..self.layers.len()).rev() {
            let (weight, bias) = &mut self.layers[i];
            let input_transpose = input.transpose();
            let mut grad_weight = grad_output.clone() * input_transpose;
            let grad_bias = grad_output.clone();
            grad_output = weight.transpose() * &grad_output;

            let mut grad_input = self.batch_norms[i].backward(input, &grad_output);

            match self.activation {
                ActivationFunction::Sigmoid => grad_input.component_mul_assign(&(output.map(|x| x * (1.0 - x)))),
                ActivationFunction::ReLU => grad_input.component_mul_assign(&(input.map(|x| if x > 0.0 { 1.0 } else { 0.0 }))),
                ActivationFunction::Tanh => grad_input.component_mul_assign(&(output.map(|x| 1.0 - x.powi(2)))),
                ActivationFunction::LeakyReLU(alpha) => grad_input.component_mul_assign(&(input.map(|x| if x >= 0.0 { 1.0 } else { alpha }))),
                ActivationFunction::ELU(alpha) => grad_input.component_mul_assign(&(input.map(|x| if x >= 0.0 { 1.0 } else { x.exp() + alpha }))),
            };

            grad_weight /= input.len() as f64;
            grad_bias /= input.len() as f64;

            let (optimizer_weight, optimizer_bias) = &mut self.optimizer_state[i];
            match self.optimizer {
                Optimizer::SGD => {
                    *weight -= grad_weight;
                    *bias -= grad_bias;
                }
                Optimizer::SGDMomentum(momentum) => {
                    *optimizer_weight *= momentum;
                    *optimizer_weight += grad_weight;
                    *weight -= optimizer_weight;
                    *optimizer_bias *= momentum;
                    *optimizer_bias += grad_bias;
                    *bias -= optimizer_bias;
                }
                Optimizer::RMSProp(decay_rate, epsilon) => {
                    *optimizer_weight *= decay_rate;
                    *optimizer_weight += (1.0 - decay_rate) * grad_weight.component_mul(&grad_weight);
                    *weight -= grad_weight.component_div(&(optimizer_weight.map(|x| x.sqrt() + epsilon)));
                    *optimizer_bias *= decay_rate;
                    *optimizer_bias += (1.0 - decay_rate) * grad_bias.component_mul(&grad_bias);
                    *bias -= grad_bias.component_div(&(optimizer_bias.map(|x| x.sqrt() + epsilon)));
                }
                Optimizer::Adam(beta1, beta2) => {
                    *optimizer_weight *= beta1;
                    *optimizer_weight += (1.0 - beta1) * grad_weight;
                    let mut corrected_weight = optimizer_weight.clone();
                    corrected_weight /= (1.0 - beta1.powi(2)) as f64;
                    *optimizer_bias *= beta1;
                    *optimizer_bias += (1.0 - beta1) * grad_bias;
                    let mut corrected_bias = optimizer_bias.clone();
                    corrected_bias /= (1.0 - beta1.powi(2)) as f64;
                    *weight -= corrected_weight.component_div(&(grad_weight.map(|x| x.abs().max(1e-8)) + epsilon));
                    *bias -= corrected_bias.component_div(&(grad_bias.map(|x| x.abs().max(1e-8)) + epsilon));
                }
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
