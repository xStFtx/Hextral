## Hextral

Hextral is a Rust library for implementing a neural network with regularization techniques such as L2 and L1 regularization.

## Features

- Implements a neural network with customizable activation functions (Sigmoid, ReLU, Tanh).
- Supports L2 and L1 regularization for controlling overfitting.
- Provides methods for training the neural network, making predictions, and evaluating performance.
- Built using the nalgebra crate for efficient linear algebra operations.

## Usage

Add this crate to your `Cargo.toml`:

```toml
[dependencies]
hextral = "0.1.0"
```

Then, you can use Hextral in your Rust project as follows:

```rust
use hextral::{Hextral, ActivationFunction, Regularization};
use nalgebra::{DVector, DMatrix};

fn main() {
    // Create a new Hextral neural network
    let mut hextral = Hextral::new(0.1, 0.2);

    // Generate training data (inputs and targets)
    let inputs = vec![
        DVector::from_iterator(10, (0..10).map(|_| rand::random::<f64>())),
        // Add more input vectors as needed
    ];

    let targets = vec![
        DVector::from_iterator(10, (0..10).map(|_| rand::random::<f64>())),
        // Add corresponding target vectors as needed
    ];

    // Train the neural network
    hextral.train(&inputs, &targets, 0.01, Regularization::L2(0.001), 100);

    // Make predictions
    let input = DVector::from_iterator(10, (0..10).map(|_| rand::random::<f64>()));
    let prediction = hextral.predict(&input);
    println!("Prediction: {:?}", prediction);

    // Evaluate the model
    let evaluation_loss = hextral.evaluate(&inputs, &targets);
    println!("Evaluation Loss: {}", evaluation_loss);
}
```

For more details on the available methods and options, please refer to the [documentation](https://docs.rs/hextral/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
