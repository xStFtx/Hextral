use hextral::*;
use nalgebra::DVector;

#[tokio::main]
async fn main() -> HextralResult<()> {
    println!("Early Stopping and Checkpointing Demo");
    println!("===================================");

    // Create some synthetic data for demonstration
    let train_inputs: Vec<DVector<f64>> = (0..100)
        .map(|i| {
            let x = i as f64 / 100.0;
            DVector::from_vec(vec![x, x * x])
        })
        .collect();

    let train_targets: Vec<DVector<f64>> = train_inputs
        .iter()
        .map(|input| DVector::from_vec(vec![input[0] * 0.5 + input[1] * 0.3]))
        .collect();

    // Create validation data
    let val_inputs: Vec<DVector<f64>> = (100..120)
        .map(|i| {
            let x = i as f64 / 100.0;
            DVector::from_vec(vec![x, x * x])
        })
        .collect();

    let val_targets: Vec<DVector<f64>> = val_inputs
        .iter()
        .map(|input| DVector::from_vec(vec![input[0] * 0.5 + input[1] * 0.3]))
        .collect();

    // Create neural network
    let mut nn = Hextral::new(
        2,
        &[8, 4],
        1,
        ActivationFunction::ReLU,
        Optimizer::adam(0.001),
    );

    println!("Training without early stopping (baseline):");
    let _ = nn.train(&train_inputs, &train_targets, 0.01, 100, None, None, None, None, None).await?;
    let baseline_val_loss = nn.evaluate(&val_inputs, &val_targets).await;
    println!("Final validation loss: {:.6}", baseline_val_loss);

    // Reset network for fair comparison
    let mut nn2 = Hextral::new(
        2,
        &[8, 4],
        1,
        ActivationFunction::ReLU,
        Optimizer::adam(0.001),
    );

    println!("\nTraining with early stopping (patience=10, min_delta=0.001):");
    let early_stop = EarlyStopping::new(10, 0.001, true);
    let checkpoint_config = CheckpointConfig::new("checkpoints/early_stopping/model_checkpoint".to_string()).save_every(20);

    let (train_history, val_history) = nn2.train(
        &train_inputs,
        &train_targets,
        0.01,
        100,
        None,
        Some(&val_inputs),
        Some(&val_targets),
        Some(early_stop),
        Some(checkpoint_config),
    ).await.unwrap();

    println!("Training stopped after {} epochs", train_history.len());
    println!("Final validation loss: {:.6}", val_history.last().unwrap_or(&0.0));
    
    println!("\nValidation loss progression:");
    for (epoch, &loss) in val_history.iter().enumerate() {
        if epoch % 10 == 0 || epoch == val_history.len() - 1 {
            println!("Epoch {}: {:.6}", epoch + 1, loss);
        }
    }

    // Demonstrate async training with early stopping
    println!("\nAsync training with early stopping:");
    let mut nn3 = Hextral::new(
        2,
        &[8, 4],
        1,
        ActivationFunction::Tanh,
        Optimizer::adamw(0.001, 0.01),
    );

    let early_stop_async = EarlyStopping::new(15, 0.0005, true);
    let checkpoint_config_async = CheckpointConfig::new("checkpoints/early_stopping/async_model_checkpoint".to_string());

    let (train_history_async, val_history_async) = nn3
        .train(
            &train_inputs,
            &train_targets,
            0.01,
            100,
            Some(16),
            Some(&val_inputs),
            Some(&val_targets),
            Some(early_stop_async),
            Some(checkpoint_config_async),
        )
        .await.unwrap();

    println!("Async training stopped after {} epochs", train_history_async.len());
    println!("Final validation loss: {:.6}", val_history_async.last().unwrap_or(&0.0));

    // Compare convergence
    println!("\nConvergence comparison:");
    println!("Baseline (100 epochs): {:.6}", baseline_val_loss);
    println!("Early stopping (sync): {:.6}", val_history.last().unwrap_or(&0.0));
    println!("Early stopping (async): {:.6}", val_history_async.last().unwrap_or(&0.0));
    
    Ok(())
}