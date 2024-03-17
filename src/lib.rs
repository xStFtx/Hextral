extern crate nalgebra;
extern crate rand;

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

pub struct Hextral {
    h: DMatrix<f64>,
    qft: f64,
    laplace: f64,
}

impl Hextral {
    pub fn new(qft: f64, laplace: f64) -> Self {
        let h = DMatrix::from_fn(10, 10, |_, _| rand::thread_rng().gen::<f64>() * 0.1);
        Hextral { h, qft, laplace }
    }

    pub fn forward_pass(&self, input: &DVector<f64>, activation: ActivationFunction) -> DVector<f64> {
        let output = &self.h * input;

        let output = match activation {
            ActivationFunction::Sigmoid => output.map(|x| sigmoid(x)),
            ActivationFunction::ReLU => output.map(|x| x.max(0.0)),
            ActivationFunction::Tanh => output.map(|x| x.tanh()),
            ActivationFunction::LeakyReLU(alpha) => output.map(|x| if x >= 0.0 { x } else { alpha * x }),
            ActivationFunction::ELU(alpha) => output.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }),
        };

        output
    }

    pub fn train(&mut self, inputs: &[DVector<f64>], targets: &[DVector<f64>], learning_rate: f64, regularization: Regularization, epochs: usize, batch_size: usize) {
        for _ in 0..epochs {
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let batch_inputs = &inputs[batch_start..batch_end];
                let batch_targets = &targets[batch_start..batch_end];

                let mut batch_gradients = DMatrix::zeros(self.h.nrows(), self.h.ncols());

                for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                    let output = self.forward_pass(input, ActivationFunction::Sigmoid);
                    let loss_gradient = &output - target;
                    let gradients = loss_gradient.clone() * input.transpose();
                    batch_gradients += gradients;
                }

                batch_gradients /= batch_size as f64;

                self.update_parameters(learning_rate, &batch_gradients, &regularization);
            }
        }
    }

    pub fn update_parameters(&mut self, learning_rate: f64, gradients: &DMatrix<f64>, regularization: &Regularization) {
        let gradient_update = learning_rate * gradients;

        match regularization {
            Regularization::L2(lambda) => {
                self.h *= 1.0 - learning_rate * *lambda;
                self.h -= &gradient_update;
            }
            Regularization::L1(lambda) => {
                let signum = self.h.map(|x| x.signum());
                self.h *= 1.0 - learning_rate * *lambda;
                self.h -= &gradient_update;
                self.h += learning_rate * *lambda * &signum;
            }
            Regularization::Dropout(rate) => {
                let dropout_mask = DMatrix::from_fn(gradients.nrows(), gradients.ncols(), |_, _| {
                    if rand::thread_rng().gen::<f64>() < *rate {
                        0.0
                    } else {
                        1.0 / (1.0 - *rate)
                    }
                });
                self.h = &self.h.component_mul(&dropout_mask) - &gradient_update;
            }
        }
    }

    pub fn predict(&self, input: &DVector<f64>) -> DVector<f64> {
        self.forward_pass(input, ActivationFunction::Sigmoid)
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
    let mut hextral = Hextral::new(0.1, 0.2);

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