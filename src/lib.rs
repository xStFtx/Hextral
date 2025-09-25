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
    /// Swish activation: f(x) = x * sigmoid(βx)
    Swish { beta: f64 },
    /// GELU activation: f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    GELU,
    /// Mish activation: f(x) = x * tanh(softplus(x))
    Mish,
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

/// Loss functions for training
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// Mean Squared Error: (1/2) * (y - ŷ)²
    MeanSquaredError,
    /// Mean Absolute Error: |y - ŷ|
    MeanAbsoluteError,
    /// Binary Cross Entropy: -(y*log(ŷ) + (1-y)*log(1-ŷ))
    BinaryCrossEntropy,
    /// Categorical Cross Entropy: -Σ(y*log(ŷ))
    CategoricalCrossEntropy,
    /// Huber Loss: smooth combination of MSE and MAE
    Huber { delta: f64 },
}

impl Default for LossFunction {
    fn default() -> Self {
        LossFunction::MeanSquaredError
    }
}

/// Batch normalization layer parameters
#[derive(Debug, Clone)]
pub struct BatchNormLayer {
    /// Scale parameter (gamma)
    gamma: DVector<f64>,
    /// Shift parameter (beta)  
    beta: DVector<f64>,
    /// Running mean for inference
    running_mean: DVector<f64>,
    /// Running variance for inference
    running_var: DVector<f64>,
    /// Momentum for updating running statistics
    momentum: f64,
    /// Small value for numerical stability
    epsilon: f64,
    /// Whether we're in training mode
    training: bool,
}

impl BatchNormLayer {
    /// Create a new batch normalization layer
    pub fn new(size: usize) -> Self {
        Self {
            gamma: DVector::from_element(size, 1.0), // Initialize to 1
            beta: DVector::zeros(size),               // Initialize to 0
            running_mean: DVector::zeros(size),
            running_var: DVector::from_element(size, 1.0),
            momentum: 0.1,
            epsilon: 1e-5,
            training: true,
        }
    }
    
    /// Apply batch normalization forward pass
    pub fn forward(&mut self, x: &DVector<f64>) -> (DVector<f64>, Option<(DVector<f64>, DVector<f64>, DVector<f64>)>) {
        if self.training {
            // Training mode: compute batch statistics
            let mean = x.mean();
            let var = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64;
            let std_dev = (var + self.epsilon).sqrt();
            
            // Normalize
            let normalized = x.map(|xi| (xi - mean) / std_dev);
            
            // Scale and shift
            let output = normalized.component_mul(&self.gamma) + &self.beta;
            
            // Update running statistics
            self.running_mean = &self.running_mean * (1.0 - self.momentum) + &DVector::from_element(x.len(), mean * self.momentum);
            self.running_var = &self.running_var * (1.0 - self.momentum) + &DVector::from_element(x.len(), var * self.momentum);
            
            // Return normalized values and cache for backward pass
            let cache = Some((normalized, DVector::from_element(x.len(), mean), DVector::from_element(x.len(), std_dev)));
            (output, cache)
        } else {
            // Inference mode: use running statistics
            let normalized = x.zip_map(&self.running_mean, |xi, mean| {
                (xi - mean) / (self.running_var[0] + self.epsilon).sqrt()
            });
            let output = normalized.component_mul(&self.gamma) + &self.beta;
            (output, None)
        }
    }
    
    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Main neural network struct
pub struct Hextral {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
    activation: ActivationFunction,
    optimizer: Optimizer,
    regularization: Regularization,
    loss_function: LossFunction,
    batch_norm_layers: Vec<Option<BatchNormLayer>>,
    use_batch_norm: bool,
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
            loss_function: LossFunction::default(),
            batch_norm_layers: Vec::new(),
            use_batch_norm: false,
        }
    }

    /// Set regularization
    pub fn set_regularization(&mut self, reg: Regularization) {
        self.regularization = reg;
    }

    /// Set loss function
    pub fn set_loss_function(&mut self, loss: LossFunction) {
        self.loss_function = loss;
    }
    
    /// Enable batch normalization for all hidden layers
    pub fn enable_batch_norm(&mut self) {
        if !self.use_batch_norm {
            self.use_batch_norm = true;
            self.batch_norm_layers.clear();
            
            // Add batch norm layers for all but the output layer
            for i in 0..self.layers.len() - 1 {
                let layer_size = self.layers[i].0.nrows(); // Number of outputs from this layer
                self.batch_norm_layers.push(Some(BatchNormLayer::new(layer_size)));
            }
            // No batch norm for output layer
            self.batch_norm_layers.push(None);
        }
    }
    
    /// Disable batch normalization
    pub fn disable_batch_norm(&mut self) {
        self.use_batch_norm = false;
        self.batch_norm_layers.clear();
    }
    
    /// Set training mode for batch normalization
    pub fn set_training_mode(&mut self, training: bool) {
        for bn_layer in &mut self.batch_norm_layers {
            if let Some(bn) = bn_layer {
                bn.set_training(training);
            }
        }
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
            ActivationFunction::Swish { beta } => {
                input.map(|x| x * sigmoid(beta * x))
            },
            ActivationFunction::GELU => {
                input.map(|x| {
                    0.5 * x * (1.0 + (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt() 
                        * (x + 0.044715 * x.powi(3)).tanh())
                })
            },
            ActivationFunction::Mish => {
                input.map(|x| x * (x.exp().ln_1p()).tanh()) // softplus(x) = ln(1 + exp(x))
            },
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
            ActivationFunction::Swish { beta } => {
                input.map(|x| {
                    let s = sigmoid(beta * x);
                    s + beta * x * s * (1.0 - s)
                })
            },
            ActivationFunction::GELU => {
                input.map(|x| {
                    let tanh_arg = (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt() 
                        * (x + 0.044715 * x.powi(3));
                    let tanh_val = tanh_arg.tanh();
                    let sech_sq = 1.0 - tanh_val.powi(2);
                    
                    0.5 * (1.0 + tanh_val) + 0.5 * x * sech_sq * 
                    (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt() * 
                    (1.0 + 3.0 * 0.044715 * x.powi(2))
                })
            },
            ActivationFunction::Mish => {
                input.map(|x| {
                    let softplus = x.exp().ln_1p();
                    let tanh_softplus = softplus.tanh();
                    let sigmoid_x = sigmoid(x);
                    
                    tanh_softplus + x * sigmoid_x * (1.0 - tanh_softplus.powi(2))
                })
            },
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

    /// Compute loss between prediction and target
    pub fn compute_loss(&self, prediction: &DVector<f64>, target: &DVector<f64>) -> f64 {
        match &self.loss_function {
            LossFunction::MeanSquaredError => {
                let error = prediction - target;
                0.5 * error.dot(&error)
            },
            LossFunction::MeanAbsoluteError => {
                let error = prediction - target;
                error.iter().map(|x| x.abs()).sum::<f64>()
            },
            LossFunction::BinaryCrossEntropy => {
                let mut loss = 0.0;
                for (pred, targ) in prediction.iter().zip(target.iter()) {
                    let p = pred.max(1e-15).min(1.0 - 1e-15); // Clamp to avoid log(0)
                    loss -= targ * p.ln() + (1.0 - targ) * (1.0 - p).ln();
                }
                loss
            },
            LossFunction::CategoricalCrossEntropy => {
                let mut loss = 0.0;
                for (pred, targ) in prediction.iter().zip(target.iter()) {
                    if *targ > 0.0 {
                        loss -= targ * pred.max(1e-15).ln();
                    }
                }
                loss
            },
            LossFunction::Huber { delta } => {
                let error = prediction - target;
                let mut loss = 0.0;
                for e in error.iter() {
                    if e.abs() <= *delta {
                        loss += 0.5 * e * e;
                    } else {
                        loss += delta * (e.abs() - 0.5 * delta);
                    }
                }
                loss
            }
        }
    }

    /// Compute loss gradient for backpropagation
    pub fn compute_loss_gradient(&self, prediction: &DVector<f64>, target: &DVector<f64>) -> DVector<f64> {
        match &self.loss_function {
            LossFunction::MeanSquaredError => {
                prediction - target
            },
            LossFunction::MeanAbsoluteError => {
                let error = prediction - target;
                error.map(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 })
            },
            LossFunction::BinaryCrossEntropy => {
                let mut grad = DVector::zeros(prediction.len());
                for i in 0..prediction.len() {
                    let p = prediction[i].max(1e-15).min(1.0 - 1e-15);
                    let t = target[i];
                    grad[i] = (p - t) / (p * (1.0 - p));
                }
                grad
            },
            LossFunction::CategoricalCrossEntropy => {
                let mut grad = DVector::zeros(prediction.len());
                for i in 0..prediction.len() {
                    if target[i] > 0.0 {
                        grad[i] = -target[i] / prediction[i].max(1e-15);
                    }
                }
                grad
            },
            LossFunction::Huber { delta } => {
                let error = prediction - target;
                error.map(|e| {
                    if e.abs() <= *delta {
                        e
                    } else {
                        delta * e.signum()
                    }
                })
            }
        }
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
        
        // Compute loss using configured loss function
        let loss = self.compute_loss(prediction, target);
        
        // Backward pass - compute loss gradient
        let mut delta = self.compute_loss_gradient(prediction, target);
        
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
            let loss = self.compute_loss(&prediction, target);
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
mod tests;