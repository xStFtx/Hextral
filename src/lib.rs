
use nalgebra::{DVector, DMatrix};
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use futures::future::join_all;
use serde::{Serialize, Deserialize};
use std::path::Path;
use tokio::fs;

pub mod activation;
pub mod optimizer;

pub use activation::ActivationFunction;
pub use optimizer::{Optimizer, OptimizerState};

#[derive(Debug, Clone)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub best_loss: f64,
    pub counter: usize,
    pub restore_best_weights: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, restore_best_weights: bool) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            counter: 0,
            restore_best_weights,
        }
    }
    
    pub fn should_stop(&mut self, current_loss: f64) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }
    
    pub fn reset(&mut self) {
        self.best_loss = f64::INFINITY;
        self.counter = 0;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub save_best: bool,
    pub save_every: Option<usize>,
    pub filepath: String,
    pub monitor_loss: bool,
}

impl CheckpointConfig {
    pub fn new<P: AsRef<Path>>(filepath: P) -> Self {
        Self {
            save_best: true,
            save_every: None,
            filepath: filepath.as_ref().to_string_lossy().to_string(),
            monitor_loss: true,
        }
    }
    
    pub fn save_every(mut self, epochs: usize) -> Self {
        self.save_every = Some(epochs);
        self
    }
    
    pub async fn save_weights(&self, weights: &[(DMatrix<f64>, DVector<f64>)]) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(weights)?;
        fs::write(&self.filepath, data).await?;
        Ok(())
    }
    
    pub async fn load_weights(&self) -> Result<Vec<(DMatrix<f64>, DVector<f64>)>, Box<dyn std::error::Error>> {
        let data = fs::read(&self.filepath).await?;
        let weights = bincode::deserialize(&data)?;
        Ok(weights)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Regularization {
    L2(f64),
    L1(f64),
    Dropout(f64),
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    /// Huber Loss: smooth combination of MSE and MAE
    Huber { delta: f64 },
}

impl Default for LossFunction {
    fn default() -> Self {
        LossFunction::MeanSquaredError
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormLayer {
    gamma: DVector<f64>,
    beta: DVector<f64>,
    running_mean: DVector<f64>,
    running_var: DVector<f64>,
    momentum: f64,
    epsilon: f64,
    training: bool,
}

impl BatchNormLayer {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: DVector::from_element(size, 1.0),
            beta: DVector::zeros(size),
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
    
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Hextral {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
    activation: ActivationFunction,
    optimizer: Optimizer,
    optimizer_state: OptimizerState,
    regularization: Regularization,
    loss_function: LossFunction,
    batch_norm_layers: Vec<Option<BatchNormLayer>>,
    use_batch_norm: bool,
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

        // Create layer shapes for optimizer state initialization
        let layer_shapes: Vec<(usize, usize)> = layers.iter()
            .map(|(w, _)| (w.nrows(), w.ncols()))
            .collect();

        Hextral {
            layers,
            activation,
            optimizer_state: OptimizerState::new(&layer_shapes),
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

    pub async fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut output = input.clone();
        
        // Only yield if network has many layers
        if self.layers.len() > 5 {
            let mid = self.layers.len() / 2;
            
            for (i, (weight, bias)) in self.layers.iter().enumerate() {
                output = weight * &output + bias;
                if i < self.layers.len() - 1 {
                    output = self.activation.apply(&output);
                }
                if i == mid {
                    tokio::task::yield_now().await;
                }
            }
        } else {
            for (i, (weight, bias)) in self.layers.iter().enumerate() {
                output = weight * &output + bias;
                if i < self.layers.len() - 1 {
                    output = self.activation.apply(&output);
                }
            }
        }
        
        output
    }

    pub async fn predict(&self, input: &DVector<f64>) -> DVector<f64> {
        self.forward(input).await
    }

    pub async fn predict_batch(&self, inputs: &[DVector<f64>]) -> Vec<DVector<f64>> {
        if inputs.len() > 10 {
            let futures: Vec<_> = inputs.iter()
                .map(|input| self.predict(input))
                .collect();
            join_all(futures).await
        } else {
            let mut results = Vec::new();
            for input in inputs {
                results.push(self.predict(input).await);
            }
            results
        }
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

    pub async fn train_step(&mut self, input: &DVector<f64>, target: &DVector<f64>, learning_rate: f64) -> f64 {
        // Forward pass - collect activations
        let mut activations = vec![input.clone()];
        let mut current = input.clone();
        
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            current = weight * &current + bias;
            if i < self.layers.len() - 1 {
                current = self.activation.apply(&current);
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
                let activation_grad = self.activation.apply_derivative(output_activation);
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
            
            // Update parameters using the new optimizer system
            let (mut weights, mut biases) = self.layers[i].clone();
            self.optimizer.update_parameters(
                &mut weights,
                &mut biases,
                &final_weight_grad,
                &bias_grad,
                &mut self.optimizer_state,
                i,
                learning_rate,
            );
            self.layers[i] = (weights, biases);
            
            // Propagate error to previous layer
            if i > 0 {
                delta = self.layers[i].0.transpose() * &delta;
            }
        }
        
        // Yield occasionally for async compatibility
        if self.layers.len() > 3 {
            tokio::task::yield_now().await;
        }
        
        loss
    }

    /// Full async training method with early stopping and checkpoints
    pub async fn train(
        &mut self,
        train_inputs: &[DVector<f64>],
        train_targets: &[DVector<f64>],
        learning_rate: f64,
        epochs: usize,
        batch_size: Option<usize>,
        val_inputs: Option<&[DVector<f64>]>,
        val_targets: Option<&[DVector<f64>]>,
        early_stopping: Option<EarlyStopping>,
        checkpoint_config: Option<CheckpointConfig>,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
        let mut train_loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut early_stop = early_stopping;
        let mut best_val_loss = f64::INFINITY;
        let batch_size = batch_size.unwrap_or(32);

        for epoch in 0..epochs {
            // Training phase
            let mut epoch_loss = 0.0;
            let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
            indices.shuffle(&mut thread_rng());
            
            for batch in indices.chunks(batch_size) {
                for &i in batch {
                    epoch_loss += self.train_step(&train_inputs[i], &train_targets[i], learning_rate).await;
                }
                if batch_size > 10 {
                    tokio::task::yield_now().await;
                }
            }
            
            let train_loss = epoch_loss / train_inputs.len() as f64;
            train_loss_history.push(train_loss);

            // Validation phase
            let val_loss = if let (Some(val_inputs), Some(val_targets)) = (val_inputs, val_targets) {
                self.evaluate(val_inputs, val_targets).await
            } else {
                train_loss // Use training loss if no validation data
            };
            val_loss_history.push(val_loss);

            // Checkpoint management
            if let Some(ref config) = checkpoint_config {
                let should_save_best = config.save_best && val_loss < best_val_loss;
                let should_save_periodic = config.save_every.map_or(false, |freq| (epoch + 1) % freq == 0);
                
                if should_save_best {
                    best_val_loss = val_loss;
                    config.save_weights(&self.layers).await?;
                }
                
                if should_save_periodic {
                    let periodic_path = format!("{}_epoch_{}", config.filepath, epoch + 1);
                    let periodic_config = CheckpointConfig::new(&periodic_path);
                    periodic_config.save_weights(&self.layers).await?;
                }
            }

            // Early stopping check
            if let Some(ref mut early_stop) = early_stop {
                if early_stop.should_stop(val_loss) {
                    if early_stop.restore_best_weights {
                        if let Some(ref config) = checkpoint_config {
                            if config.save_best {
                                match config.load_weights().await {
                                    Ok(weights) => self.set_weights(weights),
                                    Err(_) => {} // Continue with current weights if loading fails
                                }
                            }
                        }
                    }
                    break;
                }
            }

            // Yield occasionally for long training
            if epoch % 10 == 0 {
                tokio::task::yield_now().await;
            }
        }

        Ok((train_loss_history, val_loss_history))
    }

    pub async fn evaluate(&self, test_inputs: &[DVector<f64>], test_targets: &[DVector<f64>]) -> f64 {
        if test_inputs.len() > 10 {
            // Process predictions in parallel for large datasets
            let predictions = self.predict_batch(test_inputs).await;
            
            let mut total_loss = 0.0;
            for (prediction, target) in predictions.iter().zip(test_targets.iter()) {
                let loss = self.compute_loss(prediction, target);
                total_loss += loss;
            }
            total_loss / test_inputs.len() as f64
        } else {
            // Process sequentially for small datasets
            let mut total_loss = 0.0;
            for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
                let prediction = self.predict(input).await;
                let loss = self.compute_loss(&prediction, target);
                total_loss += loss;
            }
            total_loss / test_inputs.len() as f64
        }
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