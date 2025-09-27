use hextral::{ActivationFunction, Hextral, LossFunction, Optimizer};
use nalgebra::DVector;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hextral Batch Processing & Memory Optimization Demo");
    println!("=================================================");

    // Create network
    let mut nn = Hextral::new(
        4,           // input size
        &[8, 16, 8], // hidden layers
        2,           // output size (binary classification)
        ActivationFunction::ReLU,
        Optimizer::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
    );

    nn.set_loss_function(LossFunction::BinaryCrossEntropy);

    // Enable memory optimization features
    #[cfg(feature = "performance")]
    {
        nn.enable_memory_optimization(None);
        println!("Memory optimization enabled");

        // Get recommended batch size
        let recommended_batch_size = nn.recommend_batch_size(128);
        println!(
            "Recommended batch size for 128MB memory: {}",
            recommended_batch_size
        );
    }

    #[cfg(not(feature = "performance"))]
    {
        println!("Memory optimization features not available (enable 'performance' feature)");
    }

    // Generate synthetic dataset
    let dataset_size = 1000;
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();

    println!("\nGenerating dataset with {} samples...", dataset_size);

    for i in 0..dataset_size {
        let x1 = (i as f64 / 100.0).sin();
        let x2 = (i as f64 / 150.0).cos();
        let x3 = (i as f64 / 200.0).sin() * 0.5;
        let x4 = (i as f64 / 120.0).cos() * 0.5;

        let input = DVector::from_vec(vec![x1, x2, x3, x4]);

        // Binary classification: class 0 if x1 + x2 > 0, else class 1
        let label = if x1 + x2 > 0.0 { 0 } else { 1 };
        let target = if label == 0 {
            DVector::from_vec(vec![1.0, 0.0])
        } else {
            DVector::from_vec(vec![0.0, 1.0])
        };

        train_inputs.push(input);
        train_targets.push(target);
    }

    println!("Dataset generated successfully");

    // Compare standard vs optimized training
    let epochs = 50;
    let batch_size = Some(32);
    let learning_rate = 0.01;

    // Standard training
    println!("\n--- Standard Training ---");
    let start_time = Instant::now();

    let mut nn_standard = nn.clone();
    let (train_losses, _) = nn_standard
        .train(
            &train_inputs,
            &train_targets,
            learning_rate,
            epochs,
            batch_size,
            None,
            None,
            None,
            None,
        )
        .await?;

    let standard_time = start_time.elapsed();
    println!(
        "Standard training completed in {:.2}s",
        standard_time.as_secs_f32()
    );
    println!("Final training loss: {:.6}", train_losses.last().unwrap());

    // Optimized training (only if performance feature is enabled)
    #[cfg(feature = "performance")]
    {
        println!("\n--- Optimized Training ---");
        let start_time = Instant::now();

        let mut nn_optimized = nn.clone();
        let (opt_train_losses, _) = nn_optimized
            .train_optimized(
                &train_inputs,
                &train_targets,
                learning_rate,
                epochs,
                batch_size,
                None,
                None,
                None,
                None,
            )
            .await?;

        let optimized_time = start_time.elapsed();
        println!(
            "Optimized training completed in {:.2}s",
            optimized_time.as_secs_f32()
        );
        println!(
            "Final training loss: {:.6}",
            opt_train_losses.last().unwrap()
        );

        let speedup = standard_time.as_secs_f32() / optimized_time.as_secs_f32();
        println!("Speedup: {:.2}x", speedup);

        // Show memory statistics
        if let Some(stats) = nn_optimized.memory_stats() {
            println!("\nMemory Statistics:");
            println!("  Vector pools: {}", stats.vector_pools);
            println!("  Matrix pools: {}", stats.matrix_pools);
            println!("  Pool memory: {:.2} MB", stats.pool_memory_mb());
            println!("  Peak allocated: {:.2} MB", stats.peak_allocated_mb());
            println!("  Total allocations: {}", stats.allocation_count);
        }
    }

    // Batch prediction comparison
    println!("\n--- Batch Prediction Comparison ---");
    let test_size = 200;
    let test_inputs: Vec<DVector<f64>> = train_inputs.iter().take(test_size).cloned().collect();

    // Standard batch prediction
    let start_time = Instant::now();
    let standard_predictions = nn.predict_batch(&test_inputs).await;
    let standard_pred_time = start_time.elapsed();

    println!(
        "Standard batch prediction ({} samples): {:.3}s",
        test_size,
        standard_pred_time.as_secs_f32()
    );

    // Show first few predictions
    println!("First 3 predictions (standard):");
    for (i, pred) in standard_predictions.iter().take(3).enumerate() {
        println!("  Sample {}: [{:.4}, {:.4}]", i, pred[0], pred[1]);
    }

    // Optimized batch prediction
    #[cfg(feature = "performance")]
    {
        let start_time = Instant::now();
        let optimized_predictions = nn.predict_batch_optimized(&test_inputs).await?;
        let optimized_pred_time = start_time.elapsed();

        println!(
            "Optimized batch prediction ({} samples): {:.3}s",
            test_size,
            optimized_pred_time.as_secs_f32()
        );

        let pred_speedup = standard_pred_time.as_secs_f32() / optimized_pred_time.as_secs_f32();
        println!("Prediction speedup: {:.2}x", pred_speedup);

        // Verify predictions match
        let mut max_diff: f64 = 0.0;
        for (std_pred, opt_pred) in standard_predictions.iter().zip(&optimized_predictions) {
            for (a, b) in std_pred.iter().zip(opt_pred.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
        }
        println!("Max difference between predictions: {:.10}", max_diff);

        if max_diff < 1e-10 {
            println!("PASS: Predictions are identical");
        } else {
            println!("WARN: Small numerical differences detected");
        }
    }

    // Memory usage demonstration
    #[cfg(feature = "performance")]
    {
        println!("\n--- Memory Usage Demo ---");

        // Process large batch to show memory management
        let large_dataset_size = 5000;
        let mut large_inputs = Vec::new();

        for i in 0..large_dataset_size {
            let x = i as f64 / 1000.0;
            let input = DVector::from_vec(vec![x, x * 2.0, x * 3.0, x * 4.0]);
            large_inputs.push(input);
        }

        println!(
            "Processing large dataset with {} samples...",
            large_dataset_size
        );

        // Track memory before and after
        use hextral::memory::get_memory_stats;
        let initial_stats = get_memory_stats();

        let _large_predictions = nn.predict_batch_optimized(&large_inputs).await?;

        let final_stats = get_memory_stats();

        println!("Memory usage:");
        println!("  Initial: {:.2} MB", initial_stats.total_allocated_mb());
        println!("  Peak: {:.2} MB", final_stats.peak_allocated_mb());
        println!("  Final: {:.2} MB", final_stats.total_allocated_mb());
        println!(
            "  Total allocations: {}",
            final_stats.allocation_count - initial_stats.allocation_count
        );
    }

    println!("\n--- Demo completed successfully ---");

    #[cfg(not(feature = "performance"))]
    {
        println!("\nNote: To see memory optimization features, run with:");
        println!("cargo run --example batch_optimization_demo --features performance");
    }

    Ok(())
}
