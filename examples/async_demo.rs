use hextral::*;
use nalgebra::DVector;
use tokio::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hextral Async Demo");
    println!("==================\n");

    let nn = Hextral::new(
        2,
        &[8, 8],
        1,
        ActivationFunction::ReLU,
        Optimizer::adam(0.01),
    );

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

    // Compare sync vs async training
    println!("Sync vs Async Training");
    println!("----------------------");

    let mut nn_sync = nn.clone();
    let mut nn_async = nn.clone();

    let start = Instant::now();
    let _ = nn_sync
        .train(
            &train_inputs,
            &train_targets,
            0.1,
            50,
            None,
            None,
            None,
            None,
            None,
        )
        .await?;
    let sync_time = start.elapsed();

    let start = Instant::now();
    let _ = nn_async
        .train(
            &train_inputs,
            &train_targets,
            0.1,
            50,
            Some(2),
            None,
            None,
            None,
            None,
        )
        .await?;
    let async_time = start.elapsed();

    println!("First time: {:?}", sync_time);
    println!("Batched time: {:?}", async_time);

    // Batch prediction comparison
    let test_inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];

    let predictions1 = nn_sync.predict_batch(&test_inputs).await;
    let predictions2 = nn_async.predict_batch(&test_inputs).await;

    println!("\nPredictions:");
    for (i, input) in test_inputs.iter().enumerate() {
        println!(
            "[{:.0}, {:.0}] -> First: {:.3} | Batched: {:.3}",
            input[0], input[1], predictions1[i][0], predictions2[i][0]
        );
    }

    Ok(())
}
