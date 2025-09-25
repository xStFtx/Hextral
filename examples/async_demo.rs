use hextral::*;
use nalgebra::DVector;
use tokio::time::Instant;

#[tokio::main]
async fn main() {
    println!("Hextral Async Demo");
    println!("==================\n");

    let nn = Hextral::new(
        2,
        &[8, 8],
        1,
        ActivationFunction::ReLU,
        Optimizer::Adam { learning_rate: 0.01 },
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
    let _sync_loss = nn_sync.train(&train_inputs, &train_targets, 0.1, 50);
    let sync_time = start.elapsed();
    
    let start = Instant::now();
    let _async_loss = nn_async.train_async(&train_inputs, &train_targets, 0.1, 50, Some(2)).await;
    let async_time = start.elapsed();

    println!("Sync time: {:?}", sync_time);
    println!("Async time: {:?}", async_time);

    // Batch prediction comparison
    let test_inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];

    let sync_predictions = nn_sync.predict_batch(&test_inputs);
    let async_predictions = nn_async.predict_batch_async(&test_inputs).await;

    println!("\nPredictions:");
    for (i, input) in test_inputs.iter().enumerate() {
        println!("[{:.0}, {:.0}] -> Sync: {:.3} | Async: {:.3}", 
                input[0], input[1], 
                sync_predictions[i][0], 
                async_predictions[i][0]);
    }
}