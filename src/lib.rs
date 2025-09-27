use futures::future::join_all;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

pub mod activation;
pub mod error;
pub mod optimizer;

#[cfg(feature = "performance")]
pub mod batch;

#[cfg(feature = "performance")]
pub mod memory;

#[cfg(feature = "monitoring")]
pub mod monitoring;

#[cfg(feature = "datasets")]
pub mod dataset;

#[cfg(feature = "config")]
pub mod configuration;

pub use activation::ActivationFunction;
pub use error::{ContextualError, ErrorSeverity, HextralError, HextralResult, TrainingContext};
pub use optimizer::{Optimizer, OptimizerState};

#[cfg(feature = "performance")]
pub use batch::{BatchIterator, BatchProcessor, BatchStats, DataStream, MemoryPool};

#[cfg(feature = "performance")]
pub use memory::{ActivationStorage, GradientStorage, MemoryConfig, MemoryManager, MemoryStats};

#[cfg(feature = "monitoring")]
pub use monitoring::{
    ConsoleProgressCallback, EpochMetrics, PerformanceProfiler, ProgressCallback, TrainingMonitor,
    TrainingResult,
};

#[cfg(feature = "datasets")]
pub use dataset::{Dataset, DatasetError, DatasetLoader, FillStrategy, PreprocessingConfig};

#[cfg(feature = "config")]
pub use configuration::{
    BatchNormConfig, ConfigFormat, HextralBuild, HextralBuilder, HextralConfig, NetworkConfig,
    TrainingConfig,
};

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

    pub async fn save_weights(
        &self,
        weights: &[(DMatrix<f64>, DVector<f64>)],
    ) -> HextralResult<()> {
        let data = bincode::serialize(weights)?;
        fs::write(&self.filepath, data).await?;
        Ok(())
    }

    pub async fn load_weights(&self) -> HextralResult<Vec<(DMatrix<f64>, DVector<f64>)>> {
        let data = fs::read(&self.filepath).await?;
        let weights = bincode::deserialize(&data)?;
        Ok(weights)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Regularization {
    L2(f64),
    L1(f64),
    Dropout(f64),
    #[default]
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LossFunction {
    #[default]
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    /// Huber Loss: smooth combination of MSE and MAE
    Huber {
        delta: f64,
    },
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
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &mut self,
        x: &DVector<f64>,
    ) -> (
        DVector<f64>,
        Option<(DVector<f64>, DVector<f64>, DVector<f64>)>,
    ) {
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
            self.running_mean = &self.running_mean * (1.0 - self.momentum)
                + &DVector::from_element(x.len(), mean * self.momentum);
            self.running_var = &self.running_var * (1.0 - self.momentum)
                + &DVector::from_element(x.len(), var * self.momentum);

            // Return normalized values and cache for backward pass
            let cache = Some((
                normalized,
                DVector::from_element(x.len(), mean),
                DVector::from_element(x.len(), std_dev),
            ));
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

    #[cfg(feature = "performance")]
    #[serde(skip)]
    memory_manager: Option<crate::memory::MemoryManager>,

    #[cfg(feature = "performance")]
    #[serde(skip)]
    batch_processor: Option<crate::batch::BatchProcessor>,
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
            let weight = DMatrix::from_fn(size, prev_size, |_, _| rng.gen_range(-bound..bound));
            let bias = DVector::zeros(size);
            layers.push((weight, bias));
            prev_size = size;
        }

        // Initialize output layer
        let bound = (6.0 / (output_size + prev_size) as f64).sqrt();
        let weight = DMatrix::from_fn(output_size, prev_size, |_, _| rng.gen_range(-bound..bound));
        let bias = DVector::zeros(output_size);
        layers.push((weight, bias));

        // Create layer shapes for optimizer state initialization
        let layer_shapes: Vec<(usize, usize)> =
            layers.iter().map(|(w, _)| (w.nrows(), w.ncols())).collect();

        Hextral {
            layers,
            activation,
            optimizer_state: OptimizerState::new(&layer_shapes),
            optimizer,
            regularization: Regularization::None,
            loss_function: LossFunction::default(),
            batch_norm_layers: Vec::new(),
            use_batch_norm: false,

            #[cfg(feature = "performance")]
            memory_manager: None,

            #[cfg(feature = "performance")]
            batch_processor: None,
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

    /// Enable memory management optimizations
    #[cfg(feature = "performance")]
    pub fn enable_memory_optimization(&mut self, config: Option<crate::memory::MemoryConfig>) {
        let config = config.unwrap_or_default();
        self.memory_manager = Some(crate::memory::MemoryManager::new(config));
        self.batch_processor = Some(crate::batch::BatchProcessor::new(1000, 128));
    }

    /// Disable memory optimizations
    #[cfg(feature = "performance")]
    pub fn disable_memory_optimization(&mut self) {
        self.memory_manager = None;
        self.batch_processor = None;
    }

    /// Get memory statistics
    #[cfg(feature = "performance")]
    pub fn memory_stats(&self) -> Option<crate::memory::MemoryStats> {
        self.memory_manager.as_ref().map(|mm| mm.memory_stats())
    }

    /// Recommend optimal batch size based on model and memory constraints
    #[cfg(feature = "performance")]
    pub fn recommend_batch_size(&self, available_memory_mb: usize) -> usize {
        if let Some(processor) = &self.batch_processor {
            let input_size = self.layers[0].0.ncols();
            let param_count = self.parameter_count();
            processor.recommend_batch_size(input_size, available_memory_mb, param_count)
        } else {
            32 // Default batch size
        }
    }
    pub fn enable_batch_norm(&mut self) {
        if !self.use_batch_norm {
            self.use_batch_norm = true;
            self.batch_norm_layers.clear();

            // Add batch norm layers for all but the output layer
            for i in 0..self.layers.len() - 1 {
                let layer_size = self.layers[i].0.nrows(); // Number of outputs from this layer
                self.batch_norm_layers
                    .push(Some(BatchNormLayer::new(layer_size)));
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
        for bn in self.batch_norm_layers.iter_mut().flatten() {
            bn.set_training(training);
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
            let futures: Vec<_> = inputs.iter().map(|input| self.predict(input)).collect();
            join_all(futures).await
        } else {
            let mut results = Vec::new();
            for input in inputs {
                results.push(self.predict(input).await);
            }
            results
        }
    }

    /// Optimized batch prediction with memory management
    #[cfg(feature = "performance")]
    pub async fn predict_batch_optimized(
        &mut self,
        inputs: &[DVector<f64>],
    ) -> HextralResult<Vec<DVector<f64>>> {
        if self.batch_processor.is_none() {
            self.enable_memory_optimization(None);
        }

        let chunk_size = self.recommend_batch_size(256).min(inputs.len());
        let mut results = Vec::with_capacity(inputs.len());

        for chunk in inputs.chunks(chunk_size) {
            for input in chunk {
                let prediction = self.predict(input).await;
                results.push(prediction);
            }

            // Yield for large chunks
            if chunk_size > 50 {
                tokio::task::yield_now().await;
            }
        }

        Ok(results)
    }

    /// Compute loss between prediction and target
    pub fn compute_loss(&self, prediction: &DVector<f64>, target: &DVector<f64>) -> f64 {
        match &self.loss_function {
            LossFunction::MeanSquaredError => {
                let error = prediction - target;
                0.5 * error.dot(&error)
            }
            LossFunction::MeanAbsoluteError => {
                let error = prediction - target;
                error.iter().map(|x| x.abs()).sum::<f64>()
            }
            LossFunction::BinaryCrossEntropy => {
                let mut loss = 0.0;
                for (pred, targ) in prediction.iter().zip(target.iter()) {
                    let p = pred.clamp(1e-15, 1.0 - 1e-15);
                    loss -= targ * p.ln() + (1.0 - targ) * (1.0 - p).ln();
                }
                loss
            }
            LossFunction::CategoricalCrossEntropy => {
                let mut loss = 0.0;
                for (pred, targ) in prediction.iter().zip(target.iter()) {
                    if *targ > 0.0 {
                        loss -= targ * pred.max(1e-15).ln();
                    }
                }
                loss
            }
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
    pub fn compute_loss_gradient(
        &self,
        prediction: &DVector<f64>,
        target: &DVector<f64>,
    ) -> DVector<f64> {
        match &self.loss_function {
            LossFunction::MeanSquaredError => prediction - target,
            LossFunction::MeanAbsoluteError => {
                let error = prediction - target;
                error.map(|x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
            }
            LossFunction::BinaryCrossEntropy => {
                let mut grad = DVector::zeros(prediction.len());
                for i in 0..prediction.len() {
                    let p = prediction[i].clamp(1e-15, 1.0 - 1e-15);
                    let t = target[i];
                    grad[i] = (p - t) / (p * (1.0 - p));
                }
                grad
            }
            LossFunction::CategoricalCrossEntropy => {
                let mut grad = DVector::zeros(prediction.len());
                for i in 0..prediction.len() {
                    if target[i] > 0.0 {
                        grad[i] = -target[i] / prediction[i].max(1e-15);
                    }
                }
                grad
            }
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

    pub async fn train_step(
        &mut self,
        input: &DVector<f64>,
        target: &DVector<f64>,
        learning_rate: f64,
    ) -> f64 {
        // Forward pass - collect activations
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.clone());
        let mut current = input.clone();

        for (index, (weight, bias)) in self.layers.iter().enumerate() {
            current = weight * &current + bias;
            if index < self.layers.len() - 1 {
                current = self.activation.apply(&current);
            }
            activations.push(current.clone());
        }

        let prediction = activations
            .last()
            .expect("activations always contain output");

        // Compute loss using configured loss function
        let loss = self.compute_loss(prediction, target);

        // Backward pass - compute loss gradient
        let mut delta = self.compute_loss_gradient(prediction, target);

        for layer_index in (0..self.layers.len()).rev() {
            let input_activation = &activations[layer_index];
            let output_activation = &activations[layer_index + 1];

            // Apply activation derivative (except for output layer)
            if layer_index < self.layers.len() - 1 {
                let activation_grad = self.activation.apply_derivative(output_activation);
                delta = delta.component_mul(&activation_grad);
            }

            // Compute gradients
            let mut weight_grad = &delta * input_activation.transpose();
            let bias_grad = delta.clone();

            let (weights, biases) = self
                .layers
                .get_mut(layer_index)
                .expect("layer index is valid");

            match &self.regularization {
                Regularization::L2(lambda) => {
                    weight_grad += &(*weights) * *lambda;
                }
                Regularization::L1(lambda) => {
                    weight_grad += weights.map(|w| *lambda * w.signum());
                }
                Regularization::Dropout(rate) => {
                    let keep_probability = 1.0 - *rate;
                    if keep_probability <= 0.0 {
                        weight_grad.fill(0.0);
                    } else {
                        weight_grad *= keep_probability;
                    }
                }
                Regularization::None => {}
            }

            let next_delta = if layer_index > 0 {
                Some(weights.transpose() * &delta)
            } else {
                None
            };

            // Update parameters using the optimizer
            self.optimizer.update_parameters(
                weights,
                biases,
                &weight_grad,
                &bias_grad,
                &mut self.optimizer_state,
                layer_index,
                learning_rate,
            );

            if let Some(prev_delta) = next_delta {
                delta = prev_delta;
            }
        }

        // Yield occasionally for async compatibility
        if self.layers.len() > 3 {
            tokio::task::yield_now().await;
        }

        loss
    }

    /// Optimized training method with memory management and batch processing
    #[cfg(feature = "performance")]
    #[allow(clippy::too_many_arguments)]
    pub async fn train_optimized(
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
    ) -> HextralResult<(Vec<f64>, Vec<f64>)> {
        self.validate_dataset(train_inputs, train_targets, "training data")?;

        match (val_inputs, val_targets) {
            (Some(inputs), Some(targets)) => {
                self.validate_dataset(inputs, targets, "validation data")?;
            }
            (None, None) => {}
            _ => {
                return Err(HextralError::InvalidInput {
                    context: "validation data".to_string(),
                    details: "both validation inputs and targets must be provided".to_string(),
                    recoverable: false,
                });
            }
        }

        // Initialize performance optimizations if not already done
        if self.batch_processor.is_none() {
            self.enable_memory_optimization(None);
        }

        let mut train_loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut early_stop = early_stopping;
        let mut best_val_loss = f64::INFINITY;

        let effective_batch_size = batch_size
            .unwrap_or_else(|| self.recommend_batch_size(512))
            .max(1);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            // Process batches
            for batch_indices in indices.chunks(effective_batch_size) {
                let mut batch_loss = 0.0;

                // Process each sample in the batch
                for &i in batch_indices {
                    batch_loss += self
                        .train_step(&train_inputs[i], &train_targets[i], learning_rate)
                        .await;
                }

                epoch_loss += batch_loss;

                // Yield for large batches
                if effective_batch_size > 50 {
                    tokio::task::yield_now().await;
                }
            }

            let train_loss = epoch_loss / train_inputs.len() as f64;
            train_loss_history.push(train_loss);

            // Validation phase
            let val_loss = if let (Some(val_inputs), Some(val_targets)) = (val_inputs, val_targets)
            {
                self.evaluate_optimized(val_inputs, val_targets).await?
            } else {
                train_loss
            };
            val_loss_history.push(val_loss);

            // Checkpoint management
            if let Some(ref config) = checkpoint_config {
                let should_save_best = config.save_best && val_loss < best_val_loss;
                let should_save_periodic = config
                    .save_every
                    .is_some_and(|freq| (epoch + 1) % freq == 0);

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
                                if let Ok(weights) = config.load_weights().await {
                                    self.set_weights(weights);
                                }
                            }
                        }
                    }
                    break;
                }
            }

            // Memory cleanup every 10 epochs
            if epoch % 10 == 0 {
                if let Some(processor) = &mut self.batch_processor {
                    processor.clear_memory_pool();
                }
                tokio::task::yield_now().await;
            }
        }

        Ok((train_loss_history, val_loss_history))
    }

    /// Optimized evaluation method
    #[cfg(feature = "performance")]
    pub async fn evaluate_optimized(
        &mut self,
        test_inputs: &[DVector<f64>],
        test_targets: &[DVector<f64>],
    ) -> HextralResult<f64> {
        self.validate_dataset(test_inputs, test_targets, "evaluation data")?;

        if self.batch_processor.is_none() {
            self.enable_memory_optimization(None);
        }

        let chunk_size = self.recommend_batch_size(256).min(test_inputs.len());
        let mut total_loss = 0.0;

        for (chunk_inputs, chunk_targets) in test_inputs
            .chunks(chunk_size)
            .zip(test_targets.chunks(chunk_size))
        {
            for (input, target) in chunk_inputs.iter().zip(chunk_targets.iter()) {
                let prediction = self.predict(input).await;
                let loss = self.compute_loss(&prediction, target);
                total_loss += loss;
            }

            // Yield for large chunks
            if chunk_inputs.len() > 50 {
                tokio::task::yield_now().await;
            }
        }

        Ok(total_loss / test_inputs.len() as f64)
    }

    /// Full async training method with early stopping and checkpoints
    #[allow(clippy::too_many_arguments)]
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
    ) -> HextralResult<(Vec<f64>, Vec<f64>)> {
        self.validate_dataset(train_inputs, train_targets, "training data")?;

        match (val_inputs, val_targets) {
            (Some(inputs), Some(targets)) => {
                self.validate_dataset(inputs, targets, "validation data")?;
            }
            (None, None) => {}
            _ => {
                return Err(HextralError::InvalidInput {
                    context: "validation data".to_string(),
                    details: "both validation inputs and targets must be provided".to_string(),
                    recoverable: false,
                });
            }
        }

        let mut train_loss_history = Vec::new();
        let mut val_loss_history = Vec::new();
        let mut early_stop = early_stopping;
        let mut best_val_loss = f64::INFINITY;
        let batch_size = batch_size.unwrap_or(32).max(1);

        for epoch in 0..epochs {
            // Training phase
            let mut epoch_loss = 0.0;
            let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
            indices.shuffle(&mut thread_rng());

            for batch in indices.chunks(batch_size) {
                for &i in batch {
                    epoch_loss += self
                        .train_step(&train_inputs[i], &train_targets[i], learning_rate)
                        .await;
                }
                if batch_size > 10 {
                    tokio::task::yield_now().await;
                }
            }

            let train_loss = epoch_loss / train_inputs.len() as f64;
            train_loss_history.push(train_loss);

            // Validation phase
            let val_loss = if let (Some(val_inputs), Some(val_targets)) = (val_inputs, val_targets)
            {
                self.evaluate(val_inputs, val_targets).await?
            } else {
                train_loss // Use training loss if no validation data
            };
            val_loss_history.push(val_loss);

            // Checkpoint management
            if let Some(ref config) = checkpoint_config {
                let should_save_best = config.save_best && val_loss < best_val_loss;
                let should_save_periodic = config
                    .save_every
                    .is_some_and(|freq| (epoch + 1) % freq == 0);

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
                                if let Ok(weights) = config.load_weights().await {
                                    self.set_weights(weights);
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

    pub async fn evaluate(
        &self,
        test_inputs: &[DVector<f64>],
        test_targets: &[DVector<f64>],
    ) -> HextralResult<f64> {
        self.validate_dataset(test_inputs, test_targets, "evaluation data")?;

        let total_loss = if test_inputs.len() > 10 {
            // Process predictions in parallel for large datasets
            let predictions = self.predict_batch(test_inputs).await;

            predictions
                .iter()
                .zip(test_targets.iter())
                .map(|(prediction, target)| self.compute_loss(prediction, target))
                .sum()
        } else {
            // Process sequentially for small datasets
            let mut total = 0.0;
            for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
                let prediction = self.predict(input).await;
                total += self.compute_loss(&prediction, target);
            }
            total
        };

        Ok(total_loss / test_inputs.len() as f64)
    }

    fn validate_dataset(
        &self,
        inputs: &[DVector<f64>],
        targets: &[DVector<f64>],
        context: &str,
    ) -> HextralResult<()> {
        if inputs.len() != targets.len() {
            return Err(HextralError::InvalidInput {
                context: context.to_string(),
                details: format!(
                    "inputs ({}) and targets ({}) must have the same length",
                    inputs.len(),
                    targets.len()
                ),
                recoverable: false,
            });
        }

        if inputs.is_empty() {
            return Err(HextralError::InvalidInput {
                context: context.to_string(),
                details: "dataset must contain at least one sample".to_string(),
                recoverable: false,
            });
        }

        let expected_input = self.layers.first().map(|(w, _)| w.ncols()).unwrap_or(0);
        let expected_output = self.layers.last().map(|(w, _)| w.nrows()).unwrap_or(0);

        if let Some(sample) = inputs.first() {
            if sample.len() != expected_input {
                return Err(HextralError::InvalidDimensions {
                    expected: expected_input,
                    actual: sample.len(),
                });
            }
        }

        if let Some(sample) = targets.first() {
            if sample.len() != expected_output {
                return Err(HextralError::InvalidDimensions {
                    expected: expected_output,
                    actual: sample.len(),
                });
            }
        }

        Ok(())
    }

    /// Get the number of parameters in the network
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
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
