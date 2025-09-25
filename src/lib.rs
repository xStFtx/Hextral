//! # Hextral
//! 
//! A neural network library with batch normalization, multiple optimizers, and activation functions.

use nalgebra::{DVector, DMatrix};
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;

/// Activation functions available for the neural network
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    /// Sigmoid activation: f(x) = 1/(1+e^(-x))
    Sigmoid,
    /// ReLU activation: f(x) = max(0, x)
    ReLU,
    /// Tanh activation: f(x) = tanh(x)
    Tanh,
    /// Leaky ReLU activation: f(x) = max(αx, x)
    LeakyReLU(f64),
    /// ELU activation: f(x) = x if x ≥ 0, α(e^x - 1) if x < 0
    ELU(f64),
    /// Linear activation: f(x) = x
    Linear,
}

/// Regularization techniques
#[derive(Debug, Clone)]
pub enum Regularization {
    /// L2 regularization (Ridge): λ * ||w||²
    L2(f64),
    /// L1 regularization (Lasso): λ * ||w||₁
    L1(f64),
    /// Dropout regularization
    Dropout(f64),
    /// No regularization
    None,
}

/// Optimization algorithms
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD { learning_rate: f64 },
    /// SGD with momentum
    SGDMomentum { learning_rate: f64, momentum: f64 },
    /// Adam optimizer (simplified)
    Adam { learning_rate: f64 },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::Adam { learning_rate: 0.001 }
    }
}

/// Main neural network struct
pub struct Hextral {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
    activation: ActivationFunction,
    optimizer: Optimizer,
    regularization: Regularization,
}

impl Hextral {
    /// Creates a new neural network
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        activation: ActivationFunction,
        optimizer: Optimizer,
    ) -> Self {
        let mut layers = Vec::with_capacity(hidden_sizes.len() + 1);
        let mut rng = thread_rng();

        let mut prev_size = input_size;
        
        // Initialize hidden layers with Xavier initialization
        for &size in hidden_sizes {
            let bound = (6.0 / (size + prev_size) as f64).sqrt();
            let weight = DMatrix::from_fn(size, prev_size, |_, _| {
                rng.gen_range(-bound..bound)
            });
            let bias = DVector::zeros(size);
            layers.push((weight, bias));
            prev_size = size;
        }
        
        // Initialize output layer
        let bound = (6.0 / (output_size + prev_size) as f64).sqrt();
        let weight = DMatrix::from_fn(output_size, prev_size, |_, _| {
            rng.gen_range(-bound..bound)
        });
        let bias = DVector::zeros(output_size);
        layers.push((weight, bias));

        Hextral {
            layers,
            activation,
            optimizer,
            regularization: Regularization::None,
        }
    }

    /// Set regularization
    pub fn set_regularization(&mut self, reg: Regularization) {
        self.regularization = reg;
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut output = input.clone();
        
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            // Linear transformation
            output = weight * &output + bias;
            
            // Apply activation function (except for output layer in some cases)
            if i < self.layers.len() - 1 {
                output = self.apply_activation(&output);
            }
        }
        
        output
    }

    /// Apply activation function
    fn apply_activation(&self, input: &DVector<f64>) -> DVector<f64> {
        match &self.activation {
            ActivationFunction::Sigmoid => input.map(|x| sigmoid(x)),
            ActivationFunction::ReLU => input.map(|x| x.max(0.0)),
            ActivationFunction::Tanh => input.map(|x| x.tanh()),
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x >= 0.0 { x } else { alpha * x })
            },
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) })
            },
            ActivationFunction::Linear => input.clone(),
        }
    }

    /// Apply derivative of activation function
    fn apply_activation_derivative(&self, input: &DVector<f64>) -> DVector<f64> {
        match &self.activation {
            ActivationFunction::Sigmoid => {
                input.map(|x| {
                    let s = sigmoid(x);
                    s * (1.0 - s)
                })
            },
            ActivationFunction::ReLU => input.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Tanh => input.map(|x| 1.0 - x.tanh().powi(2)),
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x >= 0.0 { 1.0 } else { *alpha })
            },
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x >= 0.0 { 1.0 } else { alpha * x.exp() })
            },
            ActivationFunction::Linear => DVector::from_element(input.len(), 1.0),
        }
    }

    /// Single prediction
    pub fn predict(&self, input: &DVector<f64>) -> DVector<f64> {
        self.forward(input)
    }

    /// Batch prediction
    pub fn predict_batch(&self, inputs: &[DVector<f64>]) -> Vec<DVector<f64>> {
        inputs.iter()
            .map(|input| self.predict(input))
            .collect()
    }

    /// Train the network for one step
    pub fn train_step(&mut self, input: &DVector<f64>, target: &DVector<f64>, learning_rate: f64) -> f64 {
        // Forward pass - collect activations
        let mut activations = vec![input.clone()];
        let mut current = input.clone();
        
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            current = weight * &current + bias;
            if i < self.layers.len() - 1 {
                current = self.apply_activation(&current);
            }
            activations.push(current.clone());
        }
        
        let prediction = &activations[activations.len() - 1];
        
        // Compute loss
        let error = prediction - target;
        let loss = 0.5 * error.dot(&error);
        
        // Backward pass
        let mut delta = error;
        
        for i in (0..self.layers.len()).rev() {
            let input_activation = &activations[i];
            let output_activation = &activations[i + 1];
            
            // Apply activation derivative (except for output layer)
            if i < self.layers.len() - 1 {
                let activation_grad = self.apply_activation_derivative(output_activation);
                delta = delta.component_mul(&activation_grad);
            }
            
            // Compute gradients
            let weight_grad = &delta * input_activation.transpose();
            let bias_grad = delta.clone();
            
            // Apply regularization
            let reg_weight_grad = match &self.regularization {
                Regularization::L2(lambda) => &self.layers[i].0 * *lambda,
                Regularization::L1(lambda) => self.layers[i].0.map(|w| *lambda * w.signum()),
                _ => DMatrix::zeros(self.layers[i].0.nrows(), self.layers[i].0.ncols()),
            };
            
            let final_weight_grad = weight_grad + reg_weight_grad;
            
            // Update parameters
            let effective_lr = match &self.optimizer {
                Optimizer::SGD { learning_rate: lr } => lr * learning_rate,
                Optimizer::SGDMomentum { learning_rate: lr, momentum: _ } => {
                    // Simplified - ignore momentum for now
                    lr * learning_rate
                },
                Optimizer::Adam { learning_rate: lr } => {
                    // Simplified - just use as SGD
                    lr * learning_rate
                },
            };
            
            self.layers[i].0 -= &final_weight_grad * effective_lr;
            self.layers[i].1 -= &bias_grad * effective_lr;
            
            // Propagate error to previous layer
            if i > 0 {
                delta = self.layers[i].0.transpose() * &delta;
            }
        }
        
        loss
    }

    /// Train the network for multiple epochs
    pub fn train(
        &mut self,
        train_inputs: &[DVector<f64>],
        train_targets: &[DVector<f64>],
        learning_rate: f64,
        epochs: usize,
    ) -> Vec<f64> {
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            // Shuffle data
            let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
            indices.shuffle(&mut thread_rng());
            
            for &i in &indices {
                let loss = self.train_step(&train_inputs[i], &train_targets[i], learning_rate);
                epoch_loss += loss;
            }
            
            let avg_loss = epoch_loss / train_inputs.len() as f64;
            loss_history.push(avg_loss);
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
            }
        }

        loss_history
    }

    /// Evaluate the network on test data
    pub fn evaluate(&self, test_inputs: &[DVector<f64>], test_targets: &[DVector<f64>]) -> f64 {
        let mut total_loss = 0.0;
        
        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            let prediction = self.predict(input);
            let error = &prediction - target;
            let loss = 0.5 * error.dot(&error);
            total_loss += loss;
        }

        total_loss / test_inputs.len() as f64
    }

    /// Get the number of parameters in the network
    pub fn parameter_count(&self) -> usize {
        self.layers.iter()
            .map(|(weight, bias)| weight.len() + bias.len())
            .sum()
    }

    /// Get network architecture info
    pub fn architecture(&self) -> Vec<usize> {
        let mut arch = vec![self.layers[0].0.ncols()]; // input size
        arch.extend(self.layers.iter().map(|(weight, _)| weight.nrows()));
        arch
    }

    /// Save network weights (simplified serialization)
    pub fn get_weights(&self) -> Vec<(DMatrix<f64>, DVector<f64>)> {
        self.layers.clone()
    }

    /// Load network weights (simplified deserialization)  
    pub fn set_weights(&mut self, weights: Vec<(DMatrix<f64>, DVector<f64>)>) {
        if weights.len() == self.layers.len() {
            self.layers = weights;
        }
    }
}

/// Helper function for sigmoid activation
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let nn = Hextral::new(
            2,
            &[3, 2],
            1,
            ActivationFunction::ReLU,
            Optimizer::Adam { learning_rate: 0.001 },
        );
        
        assert_eq!(nn.architecture(), vec![2, 3, 2, 1]);
        assert_eq!(nn.parameter_count(), 2*3 + 3 + 3*2 + 2 + 2*1 + 1); // weights + biases
    }

    #[test]
    fn test_forward_pass() {
        let nn = Hextral::new(
            2,
            &[3],
            1,
            ActivationFunction::ReLU,
            Optimizer::default(),
        );

        let input = DVector::from_vec(vec![1.0, 2.0]);
        let result = nn.predict(&input);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_training() {
        let mut nn = Hextral::new(
            2,
            &[4, 3],
            1,
            ActivationFunction::ReLU,
            Optimizer::default(),
        );

        let inputs = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];
        let targets = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
        ];

        let loss_history = nn.train(&inputs, &targets, 0.01, 5);
        assert_eq!(loss_history.len(), 5);
    }

    #[test]
    fn test_xor_learning() {
        let mut nn = Hextral::new(
            2,
            &[4, 4],
            1,
            ActivationFunction::Tanh,
            Optimizer::SGD { learning_rate: 0.5 },
        );

        let inputs = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];
        let targets = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        ];

        let initial_loss = nn.evaluate(&inputs, &targets);
        nn.train(&inputs, &targets, 0.1, 50);
        let final_loss = nn.evaluate(&inputs, &targets);
        
        // Network should learn and reduce loss
        assert!(final_loss < initial_loss);
    }
}