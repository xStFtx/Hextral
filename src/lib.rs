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
    gamma: DVector<f64>,
    beta: DVector<f64>,
    running_mean: DVector<f64>,
    running_var: DVector<f64>,
    eps: f64,
}

impl BatchNormalization {
    pub fn new(size: usize, eps: f64) -> Self {
        BatchNormalization {
            gamma: DVector::repeat(size, 1.0),
            beta: DVector::zeros(size),
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
            let output = &self.gamma.component_mul(&normalized) + &self.beta;
            self.running_mean = (&self.running_mean * 0.9) + (mean * 0.1);
            self.running_var = (&self.running_var * 0.9) + (var * 0.1);
            output
        } else {
            let normalized = input.map(|x| (x - &self.running_mean) / (&self.running_var.map(|x| x.sqrt()) + self.eps));
            &self.gamma.component_mul(&normalized) + &self.beta
        }
    }

    pub fn backward(&self, input: &DVector<f64>, grad_output: &DVector<f64>) -> DVector<f64> {
        let normalized = input.map(|x| (x - &self.running_mean) / (&self.running_var.map(|x| x.sqrt()) + self.eps));
        let grad_gamma = grad_output.component_mul(&normalized);
        let grad_beta = grad_output.clone();
        let grad_input = grad_output.component_mul(&(&self.gamma / (&self.running_var.map(|x| x.sqrt()) + self.eps)));
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

    pub fn train(
        &mut self,
        inputs: &[DVector<f64>],
        targets: &[DVector<f64>],
        learning_rate: f64,
        regularization: Regularization,
        epochs: usize,
        batch_size: usize,
    ) {
        for _ in 0..epochs {
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let batch_inputs = &inputs[batch_start..batch_end];
                let batch_targets = &targets[batch_start..batch_end];

                let mut batch_grads = self.optimizer_state.iter_mut().map(|(gw, gb)| (*gw, *gb)).collect::<Vec<_>>();
                for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                    let mut outputs = Vec::with_capacity(self.layers.len() + 1);
                    outputs.push(input.clone());

                    // Forward Pass
                    let mut output = input.clone();
                    for (i, (weight, bias)) in self.layers.iter().enumerate() {
                        output = weight * &output + bias;
                        if i < self.batch_norms.len() {
                            output = self.batch_norms[i].forward(&output, true);
                        }
                        output = match self.activation {
                            ActivationFunction::Sigmoid => output.map(|x| sigmoid(x)),
                            ActivationFunction::ReLU => output.map(|x| x.max(0.0)),
                            ActivationFunction::Tanh => output.map(|x| x.tanh()),
                            ActivationFunction::LeakyReLU(alpha) => output.map(|x| if x >= 0.0 { x } else { alpha * x }),
                            ActivationFunction::ELU(alpha) => output.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }),
                        };
                        outputs.push(output.clone());
                    }

                    // Backward Pass
                    let mut grad_output = cost_gradient(&output, target);
                    for i in (0..self.layers.len()).rev() {
                        let (gw, gb) = &mut batch_grads[i];
                        let grad_input = self.batch_norms[i].backward(&outputs[i], &grad_output);
                        *gw += grad_output.clone() * &outputs[i].transpose();
                        *gb += grad_output.clone();
                        grad_output = self.layers[i].0.transpose() * &grad_input;
                    }
                }

                // Update weights and biases
                for ((weight, bias), (gw, gb)) in self.layers.iter_mut().zip(batch_grads.iter()) {
                    *weight -= learning_rate * (gw + match regularization {
                        Regularization::L2(lambda) => &*weight * lambda,
                        Regularization::L1(_) => unimplemented!(), // not implemented yet
                        Regularization::Dropout(_) => unimplemented!(), // not implemented yet
                    });
                    *bias -= learning_rate * gb;
                }
                self.optimizer_state = batch_grads;
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn cost_gradient(output: &DVector<f64>, target: &DVector<f64>) -> DVector<f64> {
    output - target
}
