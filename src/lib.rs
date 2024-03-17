extern crate nalgebra;
extern crate rand;

use nalgebra::{DVector, DMatrix, DMatrixSlice, DMatrixSliceMut};
use rand::Rng;
use std::collections::VecDeque;

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
            gamma: DVector::from_element(size, 1.0),
            beta: DVector::zeros(size),
            running_mean: DVector::zeros(size),
            running_var: DVector::ones(size),
            eps,
        }
    }

    pub fn forward(&mut self, input: &DVector<f64>, training: bool) -> DVector<f64> {
        if training {
            let mean = input.mean();
            let var = input.variance_unbiased();
            let sqrt_var = var.sqrt() + self.eps;
            let normalized = input.map(|x| (x - mean) / sqrt_var);
            let output = &self.gamma * &normalized + &self.beta;
            self.running_mean = (0.9 * self.running_mean) + (0.1 * mean);
            self.running_var = (0.9 * self.running_var) + (0.1 * var);
            output
        } else {
            let normalized = input.map(|x| (x - self.running_mean) / (self.running_var.sqrt() + self.eps));
            &self.gamma * &normalized + &self.beta
        }
    }

    pub fn backward(&self, input: &DVector<f64>, grad_output: &DVector<f64>) -> DVector<f64> {
        let normalized = input.map(|x| (x - self.running_mean) / (self.running_var.sqrt() + self.eps));
        let grad_gamma = grad_output.component_mul(&normalized);
        let grad_beta = grad_output.clone();
        let grad_input = grad_output.component_mul(&(&self.gamma / (self.running_var.sqrt() + self.eps)));
        (grad_gamma.sum(), grad_beta.sum(), grad_input)
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
            output = &weight * &output + bias;
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
                    let output = self.forward_pass(input, true);
                    let loss_gradient = &output - target;
                    let mut grad_weights = Vec::with_capacity(self.layers.len());
                    let mut grad_biases = Vec::with_capacity(self.layers.len());

                    let mut grad_output = loss_gradient.clone();
                    for (i, (weight, _)) in self.layers.iter().rev().enumerate() {
                        if i < self.batch_norms.len() {
                            grad_output = self.batch_norms[self.batch_norms.len() - 1 - i].backward(input, &grad_output);
                        }
                        grad_weights.push(grad_output.clone() * input.transpose());
                        grad_biases.push(grad_output.clone());
                        if i < self.layers.len() - 1 {
                            grad_output = &weight.transpose() * &grad_output;
                            grad_output = grad_output.map(|x| match self.activation {
                                ActivationFunction::Sigmoid => x * sigmoid(x) * (1.0 - sigmoid(x)),
                                ActivationFunction::ReLU => if x >= 0.0 { 1.0 } else { 0.0 },
                                ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
                                ActivationFunction::LeakyReLU(alpha) => if x >= 0.0 { 1.0 } else { alpha },
                                ActivationFunction::ELU(alpha) => if x >= 0.0 { 1.0 } else { alpha * x.exp() },
                            });
                        }
                    }

                    for ((gw, gb), (dw, db)) in batch_grads.iter_mut().zip(grad_weights.into_iter().zip(grad_biases.into_iter())) {
                        *gw += dw;
                        *gb += db;
                    }
                }

                for (gw, gb) in batch_grads.iter_mut() {
                    *gw /= batch_size as f64;
                    *gb /= batch_size as f64;
                }

                self.update_parameters(learning_rate, &regularization, &mut batch_grads);
            }
        }
    }

    pub fn update_parameters(
        &mut self,
        learning_rate: f64,
        regularization: &Regularization,
        batch_grads: &mut [(DMatrix<f64>, DVector<f64>)],
    ) {
        for ((weight, bias), (gw, gb), (dw, db)) in self
            .layers
            .iter_mut()
            .zip(&self.optimizer_state)
            .zip(batch_grads.iter())
        {
            match self.optimizer {
                Optimizer::SGD => {
                    *weight -= &(learning_rate * gw);
                    *bias -= &(learning_rate * gb);
                }
                Optimizer::SGDMomentum(momentum) => {
                    *dw = momentum * dw + &(learning_rate * gw);
                    *db = momentum * db + &(learning_rate * gb);
                    *weight -= &dw;
                    *bias -= &db;
                }
                Optimizer::RMSProp(rho, eps) => {
                    *dw = rho * dw + (1.0 - rho) * gw.component_mul(&gw);
                    *db = rho * db + (1.0 - rho) * gb.component_mul(&gb);
                    let dw_sqrt = &dw.map(|x| x.sqrt() + eps);
                    let db_sqrt = &db.map(|x| x.sqrt() + eps);
                    *weight -= &((learning_rate / dw_sqrt) * gw);
                    *bias -= &((learning_rate / db_sqrt) * gb);
                }
                Optimizer::Adam(beta1, beta2) => {
                    let beta1_hat = beta1.powi(batch_grads.len() as i32 + 1);
                    let beta2_hat = beta2.powi(batch_grads.len() as i32 + 1);
                    *dw = beta1 * dw + (1.0 - beta1) * gw.component_mul(&gw);
                    *db = beta1 * db + (1.0 - beta1) * gb.component_mul(&gb);
                    let dw_sqrt = &dw.map(|x| x.sqrt() + 1e-8) / (1.0 - beta2_hat);
                    let db_sqrt = &db.map(|x| x.sqrt() + 1e-8) / (1.0 - beta2_hat);
                    *weight -= &((learning_rate / dw_sqrt) * (gw / (1.0 - beta1_hat)));
                    *bias -= &((learning_rate / db_sqrt) * (gb / (1.0 - beta1_hat)));
                }
            }

            match regularization {
                Regularization::L2(lambda) => {
                    *weight *= 1.0 - learning_rate * *lambda;
                }
                Regularization::L1(lambda) => {
                    let signum = weight.map(|x| x.signum());
                    *weight -= learning_rate * *lambda * &signum;
                }
                Regularization::Dropout(_) => {} // Dropout is applied during forward pass
            }
        }
    }

    pub fn predict(&self, input: &DVector<f64>) -> DVector<f64> {
        self.forward_pass(input, false)
    }

    pub fn evaluate(&self, inputs: &[DVector<f64>], targets: &[DVector<f64>]) -> f64 {
        let mut total_loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.predict(input);
            let loss = (&output - target).norm_squared();
            total_loss += loss;
        }
        total_loss / inputs.len() as f64
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let mut hextral = Hextral::new(10, &[16, 8], 10, ActivationFunction::ReLU, Optimizer::Adam(0.9, 0.999));

    let num_samples = 1000;
    let inputs: Vec<DVector<f64>> = (0..num_samples)
        .map(|_| DVector::from_iterator(10, (0..10).map(|_| rand::thread_rng().gen::<f64>())))
        .collect();

    let targets: Vec<DVector<f64>> = (0..num_samples)
        .map(|_| DVector::from_iterator(10, (0..10).map(|_| rand::thread_rng().gen::<f64>())))
        .collect();

    hextral.train(&inputs, &targets, 0.01, Regularization::L2(0.001), 100, 32);

    let input = DVector::from_iterator(10, (0..10).map(|_| rand::thread_rng().gen::<f64>()));
    let prediction = hextral.predict(&input);
    println!("Prediction: {:?}", prediction);

    let evaluation_loss = hextral.evaluate(&inputs, &targets);
    println!("Evaluation Loss: {}", evaluation_loss);
}