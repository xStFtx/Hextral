# Hextral

A high-performance neural network library for Rust with clean async-first API, comprehensive dataset loading, advanced preprocessing, multiple optimizers, early stopping, and checkpointing capabilities.

[![Crates.io](https://img.shields.io/crates/v/hextral.svg)](https://crates.io/crates/hextral)
[![Documentation](https://docs.rs/hextral/badge.svg)](https://docs.rs/hextral)

## Features

### **Core Architecture**
- **Multi-layer perceptrons** with configurable hidden layers
- **Batch normalization** for improved training stability and convergence
- **Xavier weight initialization** for stable gradient flow
- **Flexible network topology** - specify any number of hidden layers and neurons
- **Clean async-first API** with intelligent yielding for non-blocking operations

### **Dataset Loading & Processing**
- **CSV Dataset Loader** with automatic type inference, header handling, and data preprocessing
- **Image Dataset Loader** supporting PNG, JPEG, BMP, TIFF, WebP with resizing and normalization
- **Data Preprocessing Pipeline** with normalization, standardization, one-hot encoding, and PCA
- **Missing Value Handling** with forward fill, backward fill, mean, median, and mode strategies
- **Image Augmentation** with flip, rotation, brightness, contrast, and noise adjustments
- **Async Dataset Loading** with cooperative multitasking for large datasets

### **Memory Optimization & Batch Processing**
- **Memory Pool Management** - Reuse vectors and matrices to reduce allocations
- **Smart Batch Processing** - Memory-efficient processing of large datasets
- **Streaming Data Support** - Handle datasets larger than available memory
- **Memory Usage Tracking** - Monitor and optimize memory consumption
- **Adaptive Batch Sizing** - Automatic batch size recommendations based on available memory
- **In-Place Operations** - Minimize temporary object creation in critical paths

### **Activation Functions (9 Available)**
- **ReLU** - Rectified Linear Unit (good for most cases)
- **Sigmoid** - Smooth activation for binary classification  
- **Tanh** - Hyperbolic tangent for centered outputs
- **Leaky ReLU** - Prevents dying ReLU problem
- **ELU** - Exponential Linear Unit for smoother gradients
- **Linear** - For regression output layers
- **Swish** - Modern activation with smooth derivatives
- **GELU** - Gaussian Error Linear Unit used in transformers
- **Mish** - Self-regularizing activation function
- **Quaternion** - Quaternion-based normalization for 4D data

### **Loss Functions (5 Available)**
- **Mean Squared Error (MSE)** - Standard regression loss
- **Mean Absolute Error (MAE)** - Robust to outliers
- **Binary Cross-Entropy** - Binary classification
- **Categorical Cross-Entropy** - Multi-class classification
- **Huber Loss** - Robust hybrid of MSE and MAE

### **Optimization Algorithms (12 Available)**
- **Adam** - Adaptive moment estimation (recommended for most cases)
- **AdamW** - Adam with decoupled weight decay
- **NAdam** - Nesterov-accelerated Adam
- **AdaBelief** - Adapting stepsizes by belief in observed gradients
- **Lion** - Evolved sign momentum optimizer
- **SGD** - Stochastic Gradient Descent (simple and reliable)
- **SGD with Momentum** - Accelerated gradient descent
- **RMSprop** - Root mean square propagation
- **AdaGrad** - Adaptive gradient algorithm
- **AdaDelta** - Extension of AdaGrad
- **LBFGS** - Limited-memory BFGS (quasi-Newton method)
- **Ranger** - Combination of RAdam and LookAhead

### **Advanced Training Features**
- **Early Stopping** - Automatic training termination based on validation loss
- **Checkpointing** - Save and restore model weights with bincode serialization
- **Regularization** - L1/L2 regularization and dropout support
- **Batch Training** - Configurable batch sizes for memory efficiency
- **Training Progress Tracking** - Loss history and validation monitoring
- **Dual sync/async API** for both blocking and non-blocking operations
- **Memory-Optimized Training** - Reduce memory usage with smart batch processing
- **Adaptive Batch Sizing** - Automatic batch size recommendations based on available memory

### **Production Features**
- **Error Handling System** - Comprehensive error types with recovery suggestions
- **Memory Pool Management** - Reuse objects to minimize allocation overhead
- **Memory Usage Tracking** - Monitor and optimize memory consumption in real-time
- **Large Dataset Support** - Handle datasets larger than available memory through chunking
- **Performance Monitoring** - Built-in metrics collection and training progress callbacks
- **Production Logging** - Structured logging with tracing integration

### **Async/Concurrent Processing**
- **Async training methods** with cooperative multitasking
- **Parallel batch prediction** using futures
- **Intelligent yielding** - only yields for large workloads (>1000 elements)
- **Concurrent activation function processing**
- **Performance-optimized** async implementation alongside synchronous methods

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
hextral = { version = "0.9.0", features = ["datasets"] }
nalgebra = "0.33"
tokio = { version = "1.0", features = ["full"] }  # For async features
```

### Feature Flags

Hextral uses feature flags to enable optional functionality:

- **`datasets`** - CSV and image dataset loading capabilities
- **`monitoring`** - Training progress monitoring and metrics collection  
- **`performance`** - Memory optimization and advanced batch processing
- **`config`** - Configuration file support (YAML/TOML)
- **`versioning`** - Model versioning and ONNX export support
- **`testing`** - Property-based testing and benchmarking tools
- **`full`** - All features enabled

```toml
# For production use with all optimizations
hextral = { version = "0.9.0", features = ["full"] }

# For memory-constrained environments
hextral = { version = "0.9.0", features = ["performance"] }

# Basic neural network only
hextral = "0.9.0"
```

### Dataset Loading Example

```rust
use hextral::{
    Hextral, ActivationFunction, Optimizer,
    dataset::{
        csv::CsvLoader,
        image::{ImageLoader, LabelStrategy},
        preprocessing::Preprocessor,
    }
};
use nalgebra::DVector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load CSV data
    let csv_loader = CsvLoader::new()
        .with_headers(true)
        .with_target_columns_by_name(vec!["species".to_string()]);
    
    let mut dataset = csv_loader.from_file("iris.csv").await?;
    
    // Apply preprocessing
    let preprocessor = Preprocessor::new()
        .standardize(None)  // Standardize all features
        .one_hot_encode(vec![4]);  // One-hot encode target column
    
    let stats = preprocessor.fit_transform(&mut dataset).await?;
    
    // Split data (80% train, 20% test)
    let split_index = (dataset.features.len() as f64 * 0.8) as usize;
    let (train_features, test_features) = dataset.features.split_at(split_index);
    let (train_targets, test_targets) = dataset.targets.as_ref().unwrap().split_at(split_index);
    
    // Create and train neural network
    let mut nn = Hextral::new(
        dataset.metadata.feature_count,
        &[8, 6],  // Hidden layers
        3,        // Output classes
        ActivationFunction::ReLU,
        Optimizer::adam(0.001),
    );
    
    let (train_history, _) = nn.train(
        train_features,
        train_targets,
        0.01,    // Learning rate
        100,     // Epochs
        None,    // Batch size
        None, None, None, None,  // Validation, early stopping, checkpoints
    ).await?;
    
    println!("Training completed! Final loss: {:.4}", train_history.last().unwrap());
    
    // Evaluate on test set
    let test_loss = nn.evaluate(test_features, test_targets).await;
    println!("Test loss: {:.4}", test_loss);
    
    Ok(())
}
```

### Basic Async Usage (Recommended)

```rust
use hextral::{Hextral, ActivationFunction, Optimizer, EarlyStopping, CheckpointConfig};
use nalgebra::DVector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a neural network: 2 inputs -> [4, 3] hidden -> 1 output
    let mut nn = Hextral::new(
        2,                                    // Input features
        &[4, 3],                             // Hidden layer sizes  
        1,                                    // Output size
        ActivationFunction::ReLU,             // Activation function
        Optimizer::adam(0.01),                // Modern Adam optimizer
    );

    // Training data for XOR problem
    let train_inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];
    let train_targets = vec![
        DVector::from_vec(vec![0.0]),
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![0.0]),
    ];

    // Validation data (can be same as training for demo)
    let val_inputs = train_inputs.clone();
    let val_targets = train_targets.clone();

    // Configure early stopping and checkpointing
    let early_stopping = EarlyStopping::new(10, 0.001, true);
    let checkpoint_config = CheckpointConfig::new("best_model".to_string());

    // Train the network with advanced features
    println!("Training network with early stopping...");
    let (train_history, val_history) = nn.train(
        &train_inputs,
        &train_targets,
        0.1,                           // Learning rate
        1000,                          // Max epochs
        Some(2),                       // Batch size
        Some(&val_inputs),             // Validation inputs
        Some(&val_targets),            // Validation targets
        Some(early_stopping),          // Early stopping
        Some(checkpoint_config),       // Checkpointing
    ).await?;

    println!("Training completed after {} epochs", train_history.len());
    println!("Final validation loss: {:.6}", val_history.last().unwrap_or(&0.0));

    // Make predictions
    println!("\nPredictions:");
    for (input, expected) in train_inputs.iter().zip(train_targets.iter()) {
        let prediction = nn.predict(input).await;
        println!("Input: {:?} | Expected: {:.1} | Predicted: {:.3}", 
                 input.data.as_vec(), expected[0], prediction[0]);
    }

    // Batch prediction (efficient for multiple inputs)
    let batch_predictions = nn.predict_batch(&train_inputs).await;
    
    // Evaluate performance
    let final_loss = nn.evaluate(&train_inputs, &train_targets).await;
    println!("Final loss: {:.6}", final_loss);

    Ok(())
}
```

### Memory Optimization Example

```rust
use hextral::{Hextral, ActivationFunction, Optimizer, memory::MemoryConfig};
use nalgebra::DVector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut nn = Hextral::new(
        784,                                  // MNIST input size (28x28)
        &[128, 64, 32],                      // Hidden layers
        10,                                   // 10 classes (0-9)
        ActivationFunction::ReLU,
        Optimizer::adam(0.001),
    );

    // Enable memory optimization for large datasets
    nn.enable_memory_optimization(Some(MemoryConfig {
        enable_tracking: true,
        max_pool_size: 1000,
        enable_gc_hints: true,
        memory_limit_mb: Some(512),          // Limit memory usage to 512MB
        cleanup_threshold_mb: 100,           // Clean up when using > 100MB
    }));

    // Get recommended batch size based on available memory
    let batch_size = nn.recommend_batch_size(256); // 256MB available
    println!("Recommended batch size: {}", batch_size);

    // Large dataset simulation (normally loaded from files)
    let large_dataset: Vec<DVector<f64>> = (0..10000)
        .map(|i| DVector::from_fn(784, |_, _| (i as f64 % 255.0) / 255.0))
        .collect();
    let large_targets: Vec<DVector<f64>> = (0..10000)
        .map(|i| {
            let mut target = DVector::zeros(10);
            target[i % 10] = 1.0;  // One-hot encoding
            target
        })
        .collect();

    // Use optimized training for large datasets
    let (train_losses, _) = nn.train_optimized(
        &large_dataset,
        &large_targets,
        0.001,                               // Learning rate
        50,                                  // Epochs
        Some(batch_size),                    // Adaptive batch size
        None, None,                          // No validation data
        None, None,                          // No early stopping or checkpoints
    ).await?;

    // Monitor memory usage
    if let Some(stats) = nn.memory_stats() {
        println!("Memory Statistics:");
        println!("  Peak allocated: {:.2} MB", stats.peak_allocated_mb());
        println!("  Pool memory: {:.2} MB", stats.pool_memory_mb());
        println!("  Vector pools: {}", stats.vector_pools);
        println!("  Matrix pools: {}", stats.matrix_pools);
        println!("  Total allocations: {}", stats.allocation_count);
    }

    // Use optimized batch prediction for large datasets
    let predictions = nn.predict_batch_optimized(&large_dataset[..1000]).await?;
    println!("Processed {} predictions efficiently", predictions.len());

    Ok(())
}
```

### Advanced Features

```rust
use hextral::*;

#[tokio::main] 
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create network with advanced activation function
    let mut nn = Hextral::new(
        4, &[8, 6], 2,
        ActivationFunction::Swish { beta: 1.0 },  // Modern Swish activation
        Optimizer::adamw(0.001, 0.01),            // AdamW with weight decay
    );

    // Enable batch normalization for better training stability
    nn.enable_batch_norm();
    nn.set_training_mode(true);

    // Configure regularization
    nn.set_regularization(Regularization::L2(0.001));

    let inputs = vec![/* your training data */];
    let targets = vec![/* your target data */];

    // Advanced training with all features
    let early_stop = EarlyStopping::new(
        15,      // Patience: stop if no improvement for 15 epochs
        0.0001,  // Minimum improvement threshold
        true,    // Restore best weights when stopping
    );

    let checkpoint = CheckpointConfig::new("model_checkpoint".to_string())
        .save_every(10);  // Save every 10 epochs

    let (train_losses, val_losses) = nn.train(
        &inputs, &targets,
        0.01,               // Learning rate
        500,                // Max epochs
        Some(32),           // Batch size
        Some(&inputs),      // Validation inputs
        Some(&targets),     // Validation targets
        Some(early_stop),   // Early stopping
        Some(checkpoint),   // Checkpointing
    ).await?;

    // Switch to inference mode
    nn.set_training_mode(false);
    
    Ok(())
}
```
- **Scalable architecture** - Ideal for web services and concurrent applications
- **Parallel batch processing** - Multiple predictions processed concurrently using futures

### Loss Functions

Configure different loss functions for your specific task:

```rust
use hextral::{Hextral, LossFunction, ActivationFunction, Optimizer};

let mut nn = Hextral::new(2, &[4], 1, ActivationFunction::ReLU, Optimizer::default());

// For regression tasks
nn.set_loss_function(LossFunction::MeanSquaredError);
nn.set_loss_function(LossFunction::MeanAbsoluteError);
nn.set_loss_function(LossFunction::Huber { delta: 1.0 });

// For classification tasks
nn.set_loss_function(LossFunction::BinaryCrossEntropy);
nn.set_loss_function(LossFunction::CategoricalCrossEntropy);
```

### Batch Normalization

Enable batch normalization for improved training stability:

```rust
use hextral::{Hextral, ActivationFunction, Optimizer};

let mut nn = Hextral::new(10, &[64, 32], 1, ActivationFunction::ReLU, Optimizer::default());

// Enable batch normalization
nn.enable_batch_norm();

// Set training mode
nn.set_training_mode(true);

// Train your network...
let loss_history = nn.train(&inputs, &targets, 0.01, 100);

// Switch to inference mode
nn.set_training_mode(false);

// Make predictions...
let prediction = nn.predict(&input);
```

### Modern Activation Functions

Use state-of-the-art activation functions:

```rust
use hextral::{Hextral, ActivationFunction, Optimizer};

// Swish activation (used in EfficientNet)
let mut nn = Hextral::new(2, &[4], 1, 
    ActivationFunction::Swish { beta: 1.0 }, Optimizer::default());

// GELU activation (used in BERT, GPT)
let mut nn = Hextral::new(2, &[4], 1, 
    ActivationFunction::GELU, Optimizer::default());

// Mish activation (self-regularizing)
let mut nn = Hextral::new(2, &[4], 1, 
    ActivationFunction::Mish, Optimizer::default());
```


### Regularization

Prevent overfitting with built-in regularization techniques:

```rust
use hextral::{Hextral, Regularization, ActivationFunction, Optimizer};

let mut nn = Hextral::new(3, &[16, 8], 1, ActivationFunction::ReLU, 
                          Optimizer::Adam { learning_rate: 0.01 });

// L2 regularization (Ridge)
nn.set_regularization(Regularization::L2(0.01));

// L1 regularization (Lasso) 
nn.set_regularization(Regularization::L1(0.005));

// Dropout regularization
nn.set_regularization(Regularization::Dropout(0.3));
```

### Different Optimizers

Choose the optimizer that works best for your problem:

```rust
// Adam: Good default choice, adaptive learning rates
let optimizer = Optimizer::Adam { learning_rate: 0.001 };

// SGD: Simple and interpretable
let optimizer = Optimizer::SGD { learning_rate: 0.1 };

// SGD with Momentum: Accelerated convergence
let optimizer = Optimizer::SGDMomentum { 
    learning_rate: 0.1, 
    momentum: 0.9 
};
```

### Network Introspection

Get insights into your network:

```rust
// Network architecture
println!("Architecture: {:?}", nn.architecture()); // [2, 4, 3, 1]

// Parameter count  
println!("Total parameters: {}", nn.parameter_count()); // 25

// Save/load weights
let weights = nn.get_weights();
nn.set_weights(weights);
```

## Error Handling

Handle training errors gracefully with comprehensive error reporting:

```rust
use hextral::{Hextral, ActivationFunction, Optimizer, HextralResult};

#[tokio::main]
async fn main() -> HextralResult<()> {
    let mut nn = Hextral::new(2, &[8, 6], 1, ActivationFunction::ReLU, Optimizer::adam(0.01));
    
    match nn.train(&inputs, &targets, 0.01, 100, None, None, None, None, None).await {
        Ok((train_losses, val_losses)) => {
            println!("Training completed! Final loss: {:.6}", train_losses.last().unwrap());
        }
        Err(error) => {
            // Comprehensive error handling with context
            println!("Training failed: {}", error);
            println!("Severity: {:?}", error.severity());
            println!("Recoverable: {}", error.is_recoverable());
            
            if error.is_recoverable() {
                println!("Recovery suggestions:");
                for suggestion in error.recovery_suggestions() {
                    println!("  • {}", suggestion);
                }
            }
            
            return Err(error);
        }
    }
    
    Ok(())
}
```

## Dataset Loading & Preprocessing

Hextral provides comprehensive dataset loading and preprocessing capabilities.

### CSV Dataset Loading

```rust
use hextral::dataset::csv::{CsvLoader, TargetColumns};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load CSV with automatic type inference
    let csv_loader = CsvLoader::new()
        .with_headers(true)
        .with_delimiter(b',')
        .with_target_columns_by_name(vec!["target".to_string()])
        .with_max_rows(Some(1000));
    
    let dataset = csv_loader.from_file("data.csv").await?;
    
    println!("Loaded {} samples with {} features", 
             dataset.metadata.sample_count, 
             dataset.metadata.feature_count);
    
    Ok(())
}
```

### Image Dataset Loading

```rust
use hextral::dataset::image::{ImageLoader, LabelStrategy, AugmentationConfig};
use image::imageops::FilterType;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure image preprocessing and augmentation
    let augmentation = AugmentationConfig::new()
        .with_horizontal_flip(0.5)
        .with_rotation(15.0)
        .with_brightness(0.8, 1.2)
        .with_contrast(0.8, 1.2)
        .with_noise(0.1);
    
    let image_loader = ImageLoader::new()
        .with_target_size(224, 224)
        .with_normalization(true)
        .with_grayscale(false)
        .with_label_strategy(LabelStrategy::FromDirectory)
        .with_augmentation(augmentation)
        .with_extensions(vec!["jpg".to_string(), "png".to_string()]);
    
    let dataset = image_loader.from_directory("./images/").await?;
    
    println!("Loaded {} images", dataset.metadata.sample_count);
    if let Some(ref class_names) = dataset.target_names {
        println!("Classes: {:?}", class_names);
    }
    
    Ok(())
}
```

### Advanced Label Extraction

```rust
// Extract labels from directory structure
let strategy = LabelStrategy::FromDirectory;

// Extract labels from filename patterns
let strategy = LabelStrategy::FromFilename("digit".to_string());      // Extract first digit
let strategy = LabelStrategy::FromFilename("number".to_string());     // Extract first number
let strategy = LabelStrategy::FromFilename("split:_".to_string());    // Split by underscore
let strategy = LabelStrategy::FromFilename("prefix:3".to_string());   // First 3 characters

// Use manual label mapping
let mut mapping = std::collections::HashMap::new();
mapping.insert("cat_image".to_string(), 0);
mapping.insert("dog_image".to_string(), 1);
let strategy = LabelStrategy::Manual(mapping);

// Load labels from separate file
let strategy = LabelStrategy::FromFile(PathBuf::from("labels.txt"));
```

### Data Preprocessing Pipeline

```rust
use hextral::dataset::{
    preprocessing::{Preprocessor, PreprocessingUtils},
    FillStrategy
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dataset = /* load your dataset */;
    
    // Create preprocessing pipeline
    let preprocessor = Preprocessor::new()
        .normalize(None)                          // Normalize all features to [0,1]
        .standardize(Some(vec![0, 1, 2]))        // Standardize specific features
        .fill_missing(FillStrategy::Mean)         // Fill missing values with mean
        .remove_outliers(3.0)                    // Remove outliers beyond 3 std devs
        .one_hot_encode(vec![3])                 // One-hot encode categorical features
        .apply_polynomial_features(2);            // Add polynomial features (degree 2)
    
    // Fit preprocessor and transform data
    let stats = preprocessor.fit_transform(&mut dataset).await?;
    
    // Split dataset
    let (train_set, val_set, test_set) = PreprocessingUtils::train_val_test_split(
        &dataset, 0.7, 0.2  // 70% train, 20% val, 10% test
    ).await?;
    
    // Shuffle dataset
    PreprocessingUtils::shuffle(&mut dataset).await?;
    
    // Calculate correlation matrix
    let correlation = PreprocessingUtils::correlation_matrix(&dataset).await?;
    
    Ok(())
}
```

### Principal Component Analysis (PCA)

```rust
// Apply PCA for dimensionality reduction
let preprocessor = Preprocessor::new()
    .standardize(None)  // Always standardize before PCA
    .apply_pca(10);     // Reduce to 10 principal components

let stats = preprocessor.fit_transform(&mut dataset).await?;

// Features are now transformed to principal components
println!("Reduced from {} to {} dimensions", 
         stats.feature_means.len(), 
         dataset.metadata.feature_count);
```

### Missing Value Handling

```rust
use hextral::dataset::FillStrategy;

// Different strategies for handling missing values
let preprocessor = Preprocessor::new()
    .fill_missing(FillStrategy::Mean)           // Use column mean
    .fill_missing(FillStrategy::Median)         // Use column median  
    .fill_missing(FillStrategy::Mode)           // Use most frequent value
    .fill_missing(FillStrategy::Constant(0.0))  // Fill with constant
    .fill_missing(FillStrategy::ForwardFill)    // Use previous valid value
    .fill_missing(FillStrategy::BackwardFill);  // Use next valid value
```

## API Reference

### Core Types

- **`Hextral`** - Main neural network struct with async-first API
- **`ActivationFunction`** - Enum for activation functions (9 available)
- **`Optimizer`** - Enum for optimization algorithms (12 available)
- **`Regularization`** - Enum for regularization techniques
- **`EarlyStopping`** - Configuration for automatic training termination
- **`CheckpointConfig`** - Configuration for model checkpointing
- **`LossFunction`** - Enum for loss functions (5 available)

### Primary Methods (All Async)

```rust
// Network creation
Hextral::new(inputs, hidden_layers, outputs, activation, optimizer) -> Hextral

// Training with full feature set
async fn train(
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
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>>

// Predictions
async fn predict(&self, input: &DVector<f64>) -> DVector<f64>
async fn predict_batch(&self, inputs: &[DVector<f64>]) -> Vec<DVector<f64>>

// Evaluation
async fn evaluate(&self, inputs: &[DVector<f64>], targets: &[DVector<f64>]) -> f64

// Forward pass
async fn forward(&self, input: &DVector<f64>) -> DVector<f64>
```

### Configuration Methods

```rust
// Batch normalization
fn enable_batch_norm(&mut self)
fn disable_batch_norm(&mut self)
fn set_training_mode(&mut self, training: bool)

// Regularization
fn set_regularization(&mut self, reg: Regularization)

// Loss function
fn set_loss_function(&mut self, loss: LossFunction)

// Weight management
fn get_weights(&self) -> Vec<(DMatrix<f64>, DVector<f64>)>
fn set_weights(&mut self, weights: Vec<(DMatrix<f64>, DVector<f64>)>)
fn parameter_count(&self) -> usize
```

### Early Stopping & Checkpointing

```rust
// Early stopping configuration
let early_stop = EarlyStopping::new(
    patience: usize,           // Epochs to wait for improvement
    min_delta: f64,           // Minimum improvement threshold
    restore_best_weights: bool // Whether to restore best weights
);

// Checkpoint configuration  
let checkpoint = CheckpointConfig::new("model_path".to_string())
    .save_every(10)           // Save every N epochs
    .save_best(true);         // Save best model based on validation loss
```

## Performance Tips

1. **Enable performance features** with `features = ["performance"]` for production use
2. **Use recommended batch sizes** with `nn.recommend_batch_size()` for optimal memory usage
3. **Enable memory optimization** with `nn.enable_memory_optimization()` for large datasets
4. **Monitor memory usage** with `nn.memory_stats()` to track resource consumption
5. **Use ReLU activation** for hidden layers in most cases
6. **Start with Adam optimizer** - it adapts learning rates automatically
7. **Apply L2 regularization** if you see overfitting (test loss > train loss)
8. **Use dropout for large networks** to prevent co-adaptation
9. **Normalize your input data** to [0,1] or [-1,1] range for better training stability

### Memory Optimization Best Practices

- **Large Datasets**: Use `train_optimized()` and `predict_batch_optimized()` for datasets > 1000 samples
- **Memory Constraints**: Set memory limits with `MemoryConfig` to prevent OOM errors
- **Batch Processing**: Process data in chunks using adaptive batch sizing
- **Memory Pools**: Enable object reuse to minimize allocation overhead
- **Real-time Monitoring**: Track memory usage patterns for optimization opportunities

## Architecture Decisions

- **Built on nalgebra** for efficient linear algebra operations
- **Xavier initialization** for stable gradient flow from the start
- **Proper error handling** throughout the API
- **Modular design** allowing easy extension of activation functions and optimizers
- **Zero-copy predictions** where possible for performance

## Contributing

We welcome contributions! Please feel free to:

- Report bugs by opening an issue
- Suggest new features or improvements  
- Submit pull requests with enhancements
- Improve documentation
- Add more test cases

## Changelog

### v0.9.0 (Latest)

- **Configuration Management System**: YAML/TOML configuration files with environment variable support
- **Advanced Model Versioning**: Model versioning with backward compatibility and metadata storage
- **Enhanced Testing Infrastructure**: Comprehensive test suite with property-based testing and benchmarks
- **Async Runtime Optimization**: Custom executors with backpressure handling and streaming data processing
- **Enterprise Features**: Distributed training support and model serving capabilities
- **ONNX Export Support**: Export trained models to ONNX format for cross-platform deployment
- **Model Integrity Checks**: Validation and integrity verification for model files
- **Builder Pattern Architecture**: Fluent API for complex network architecture configuration

### v0.8.0

- **Complete Dataset Loading System**: CSV and image dataset loaders with async-first API
- **Comprehensive Data Preprocessing**: Normalization, standardization, one-hot encoding with dynamic category discovery
- **Advanced Missing Value Handling**: Forward/backward fill, mean, median, mode, and constant strategies
- **Principal Component Analysis (PCA)**: Full PCA implementation with power iteration and matrix deflation
- **Image Data Augmentation**: Flip, rotation, brightness, contrast, and noise adjustments with proper pixel manipulation
- **Advanced Label Extraction**: Multiple strategies for filename patterns, directory structure, and manual mapping
- **Outlier Detection and Removal**: Statistical outlier removal using IQR method with configurable thresholds
- **Polynomial Feature Engineering**: Automated polynomial feature expansion for improved model capacity
- **Memory Optimization System**: Memory pools, object reuse, and in-place operations to minimize allocations
- **Batch Processing Optimization**: Smart batch processing with streaming support for large datasets
- **Production Error Handling**: Comprehensive error types with recovery suggestions and severity classification
- **Performance Monitoring**: Built-in metrics collection, memory tracking, and training progress callbacks
- **Organized Checkpoint System**: Structured checkpoint storage with proper .gitignore configuration

### v0.7.0

- **Removed Redundancy**: Eliminated confusing duplicate methods and verbose naming patterns
- **Better Performance**: Streamlined async implementation with intelligent yielding
- **Updated Documentation**: All examples now use clean, consistent API
- **All Tests Updated**: Comprehensive test suite updated for new API patterns

### v0.6.0

- **Full Async/Await Support**: Complete async API alongside synchronous methods
- **Intelligent Yielding**: Performance-optimized async with yielding only for large workloads (>1000 elements)
- **Concurrent Processing**: Parallel batch predictions using futures and join_all
- **Async Training**: Non-blocking training with cooperative multitasking
- **Performance Improvements**: Smart async yielding prevents unnecessary overhead
- **Enhanced Documentation**: Updated examples and API documentation

### v0.5.1

- **Improved Documentation**: Enhanced README with comprehensive examples of all new features
- **Better Crates.io Presentation**: Updated documentation to properly showcase library capabilities

### v0.5.0

- **Major Feature Expansion**: Added comprehensive loss functions, batch normalization, and modern activation functions
- **5 Loss Functions**: MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber Loss
- **Batch Normalization**: Full implementation with training/inference modes
- **3 New Activation Functions**: Swish, GELU, Mish (total of 9 activation functions)
- **Code Organization**: Separated tests into dedicated files for cleaner library structure
- **Enhanced API**: Flexible loss function configuration and batch normalization controls

### v0.4.0

- **Complete rewrite** with proper error handling and fixed implementations
- **Implemented all documented features** - train(), predict(), evaluate() methods
- **Fixed critical bugs** in batch normalization and backward pass
- **Added regularization support** - L1, L2, and Dropout
- **Improved documentation** with usage examples and API reference

## License

This project is licensed under the MIT OR Apache-2.0 license.
 

