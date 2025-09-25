# Hextral

A high-performance neural network library for Rust, featuring comprehensive support for multi-layer perceptrons with advanced optimization techniques, regularization methods, and flexible architecture design.

[![Crates.io](https://img.shields.io/crates/v/hextral.svg)](https://crates.io/crates/hextral)
[![Documentation](https://docs.rs/hextral/badge.svg)](https://docs.rs/hextral)
[![License](https://img.shields.io/crates/l/hextral.svg)](LICENSE)

## Features

### **Core Architecture**
- **Multi-layer perceptrons** with configurable hidden layers
- **Batch processing** for efficient training on large datasets
- **Xavier weight initialization** for stable gradient flow
- **Flexible network topology** - specify any number of hidden layers and neurons

### **Activation Functions**
- **ReLU** - Rectified Linear Unit (default, good for most cases)
- **Sigmoid** - Smooth activation for binary classification  
- **Tanh** - Hyperbolic tangent for centered outputs
- **Leaky ReLU** - Prevents dying ReLU problem
- **ELU** - Exponential Linear Unit for smoother gradients
- **Linear** - For regression output layers

### **Optimization Algorithms**
- **Adam** - Adaptive moment estimation (recommended for most cases)
- **SGD** - Stochastic Gradient Descent (simple and reliable)
- **SGD with Momentum** - Accelerated gradient descent

### **Regularization Techniques**
- **L2 Regularization** - Prevents overfitting by penalizing large weights
- **L1 Regularization** - Encourages sparse networks and feature selection
- **Dropout** - Randomly deactivates neurons during training to prevent co-adaptation

### **Training & Evaluation**
- **Training progress tracking** with loss history
- **Batch and single-sample prediction**
- **Model evaluation** metrics and loss computation
- **Architecture introspection** - query network structure and parameter count

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
hextral = "0.4.0"
nalgebra = "0.33"
```

### Basic Usage

```rust
use hextral::{Hextral, ActivationFunction, Optimizer};
use nalgebra::DVector;

fn main() {
    // Create a neural network: 2 inputs -> [4, 3] hidden -> 1 output
    let mut nn = Hextral::new(
        2,                                    // Input features
        &[4, 3],                             // Hidden layer sizes  
        1,                                    // Output size
        ActivationFunction::ReLU,             // Activation function
        Optimizer::Adam { learning_rate: 0.01 }, // Optimizer
    );

    // Training data for XOR problem
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

    // Train the network
    println!("Training network...");
    let loss_history = nn.train(&inputs, &targets, 1.0, 100);

    // Make predictions
    for (input, expected) in inputs.iter().zip(targets.iter()) {
        let prediction = nn.predict(input);
        println!("Input: {:?} | Expected: {:.1} | Predicted: {:.3}", 
                 input.data.as_vec(), expected[0], prediction[0]);
    }

    // Evaluate performance
    let final_loss = nn.evaluate(&inputs, &targets);
    println!("Final loss: {:.6}", final_loss);
}
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

## API Reference
```

## API Reference

### Core Types

- **`Hextral`** - Main neural network struct
- **`ActivationFunction`** - Enum for activation functions
- **`Optimizer`** - Enum for optimization algorithms  
- **`Regularization`** - Enum for regularization techniques

### Key Methods

- **`new()`** - Create a new neural network
- **`train()`** - Train the network for multiple epochs
- **`predict()`** - Make a single prediction
- **`evaluate()`** - Compute loss on a dataset
- **`set_regularization()`** - Configure regularization

## Performance Tips

1. **Use ReLU activation** for hidden layers in most cases
2. **Start with Adam optimizer** - it adapts learning rates automatically
3. **Apply L2 regularization** if you see overfitting (test loss > train loss)
4. **Use dropout for large networks** to prevent co-adaptation
5. **Normalize your input data** to [0,1] or [-1,1] range for better training stability

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.4.0 (Latest)
- **Complete rewrite** with proper error handling and fixed implementations
- **Implemented all documented features** - train(), predict(), evaluate() methods
- **Fixed critical bugs** in batch normalization and backward pass
- **Added regularization support** - L1, L2, and Dropout
- **Improved documentation** with usage examples and API reference

### v0.3.5 (Previous)
- Basic neural network functionality
- Limited optimizer support
- Minimal documentation
