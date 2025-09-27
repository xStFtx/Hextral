use hextral::*;
use nalgebra::DVector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hextral Optimizer Comparison Demo");
    println!("=================================\n");

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

    // Test different optimizers
    let optimizers = vec![
        ("SGD", Optimizer::sgd(0.5)),
        ("SGD + Momentum", Optimizer::sgd_momentum(0.1, 0.9)),
        ("Adam", Optimizer::adam(0.01)),
        ("AdamW", Optimizer::adamw(0.01, 0.01)),
        ("RMSprop", Optimizer::rmsprop(0.01)),
        ("AdaGrad", Optimizer::adagrad(0.1)),
        ("NAdam", Optimizer::nadam(0.01)),
        ("Lion", Optimizer::lion(0.01)),
        ("AdaBelief", Optimizer::adabelief(0.01)),
    ];

    for (name, optimizer) in optimizers {
        println!("Testing {}", name);
        println!("{}", "-".repeat(name.len() + 8));

        let mut nn = Hextral::new(2, &[8, 8], 1, ActivationFunction::ReLU, optimizer);

        // Train for a reasonable number of epochs
        let epochs = if name.contains("AdaGrad") { 50 } else { 100 };
        let (loss_history, _) = nn
            .train(&inputs, &targets, 1.0, epochs, None, None, None, None, None)
            .await?;

        // Show final loss and predictions
        let final_loss = loss_history.last().unwrap_or(&0.0);
        println!("Final loss: {:.6}", final_loss);

        println!("Predictions:");
        for (i, input) in inputs.iter().enumerate() {
            let prediction = nn.predict(input).await;
            let expected = targets[i][0];
            let predicted = prediction[0];
            println!(
                "  [{:.0}, {:.0}] -> Expected: {:.0}, Predicted: {:.3}, Error: {:.3}",
                input[0],
                input[1],
                expected,
                predicted,
                (expected - predicted as f64).abs()
            );
        }
        println!();
    }

    println!("Demo completed! Different optimizers show different convergence patterns.");
    println!("Generally: Adam/AdamW are robust, SGD+Momentum is stable, Lion is efficient,");
    println!("RMSprop adapts learning rates, and specialized optimizers like NAdam/AdaBelief");
    println!("can provide better convergence in specific scenarios.");

    Ok(())
}
