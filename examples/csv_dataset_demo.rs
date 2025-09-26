use hextral::{
    Hextral, ActivationFunction, Optimizer, HextralResult,
    dataset::{
        csv::CsvLoader,
        preprocessing::Preprocessor,
        FillStrategy
    }
};
use nalgebra::DVector;

#[tokio::main]
async fn main() -> HextralResult<()> {
    println!("Hextral CSV Dataset Loading Example");
    println!("=====================================");

    // Example 1: Load CSV with automatic configuration
    println!("\nExample 1: Loading Iris dataset (supervised learning)");
    
    // Create sample CSV data for demonstration
    let iris_data = create_sample_iris_csv().await?;
    
    let csv_loader = CsvLoader::new()
        .with_headers(true)
        .with_last_n_targets(1) // Last column is the species (target)
        .with_max_rows(150);
    
    let dataset = csv_loader.from_string(&iris_data).await?;
    
    println!("Loaded dataset:");
    println!("   Features: {} samples × {} dimensions", 
             dataset.metadata.sample_count, 
             dataset.metadata.feature_count);
    
    if let Some(target_count) = dataset.metadata.target_count {
        println!("   Targets: {} classes", target_count);
    }
    
    // Display dataset statistics
    let stats = dataset.describe();
    println!("   Statistics computed for {} features", stats.feature_count);
    
    // Example 2: Preprocessing pipeline
    println!("\nExample 2: Data preprocessing");
    
    let mut train_dataset = dataset.clone();
    
    let preprocessor = Preprocessor::new()
        .standardize(None) // Standardize all features
        .fill_missing(FillStrategy::Mean);
    
    let preprocessing_stats = preprocessor.fit_transform(&mut train_dataset).await?;
    println!("Applied preprocessing:");
    println!("   Computed statistics for {} features", preprocessing_stats.feature_means.len());
    println!("   Standardized features to zero mean, unit variance");
    
    // Example 3: Train/validation split
    println!("\nExample 3: Data splitting");
    
    let (train_set, test_set) = train_dataset.train_test_split(0.8);
    println!("Split dataset:");
    println!("   Training: {} samples", train_set.metadata.sample_count);
    println!("   Testing: {} samples", test_set.metadata.sample_count);
    
    // Example 4: Neural network training
    println!("\nExample 4: Training neural network");
    
    let mut nn = Hextral::new(
        4,  // 4 input features (sepal/petal length & width)
        &[8, 6], // Hidden layers
        3,  // 3 output classes (setosa, versicolor, virginica)
        ActivationFunction::ReLU,
        Optimizer::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
    );
    
    // Convert dataset to vectors for training
    let train_inputs = &train_set.features;
    let train_targets = train_set.targets.as_ref().unwrap();
    
    // Convert single target values to one-hot encoded vectors
    let one_hot_targets: Vec<DVector<f64>> = train_targets.iter()
        .map(|target| {
            let class_id = target[0] as usize;
            let mut one_hot = vec![0.0; 3];
            if class_id < 3 {
                one_hot[class_id] = 1.0;
            }
            DVector::from_vec(one_hot)
        })
        .collect();
    
    println!("Training network...");
    let result = nn.train(
        train_inputs,
        &one_hot_targets,
        0.01,
        100,
        Some(16), // Batch size
        None,     // No validation data
        None,     // No validation targets
        None,     // No early stopping
        None,     // No checkpoints
    ).await;
    
    match result {
        Ok((train_losses, _val_losses)) => {
            println!("Training completed!");
            println!("   Final loss: {:.4}", train_losses.last().unwrap_or(&0.0));
        }
        Err(e) => {
            println!("Training failed: {}", e);
            return Err(e);
        }
    }
    
    // Example 5: Model evaluation
    println!("\nExample 5: Model evaluation");
    
    let test_inputs = &test_set.features;
    let test_targets = test_set.targets.as_ref().unwrap();
    
    let mut correct_predictions = 0;
    let total_predictions = test_inputs.len();
    
    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let prediction = nn.predict(input).await;
        
        // Find predicted class (highest probability)
        let predicted_class = prediction.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let actual_class = target[0] as usize;
        
        if predicted_class == actual_class {
            correct_predictions += 1;
        }
    }
    
    let accuracy = correct_predictions as f64 / total_predictions as f64;
    println!("Test accuracy: {:.2}% ({}/{} correct)", 
             accuracy * 100.0, correct_predictions, total_predictions);
    
    // Example 6: CSV with custom configuration
    println!("\nExample 6: Custom CSV configuration");
    
    let custom_data = create_sample_regression_csv().await?;
    
    let custom_loader = CsvLoader::new()
        .with_headers(true)
        .with_delimiter(b';') // Semicolon delimiter
        .with_target_names(vec!["price".to_string()])
        .with_skip_columns(vec!["id".to_string()]); // Skip ID column
    
    let regression_dataset = custom_loader.from_string(&custom_data).await?;
    
    println!("Loaded custom dataset:");
    println!("   Features: {} × {}", 
             regression_dataset.metadata.sample_count, 
             regression_dataset.metadata.feature_count);
    
    if let Some(feature_names) = &regression_dataset.feature_names {
        println!("   Feature names: {:?}", feature_names);
    }
    
    println!("\nCSV loading examples completed successfully!");
    Ok(())
}

/// Create sample Iris dataset in CSV format
async fn create_sample_iris_csv() -> Result<String, Box<dyn std::error::Error>> {
    let csv_content = r#"sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4.0,1.3,1
6.5,2.8,4.6,1.5,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
7.1,3.0,5.9,2.1,2
6.3,2.9,5.6,1.8,2
6.5,3.0,5.8,2.2,2"#;
    
    Ok(csv_content.to_string())
}

/// Create sample regression dataset in CSV format
async fn create_sample_regression_csv() -> Result<String, Box<dyn std::error::Error>> {
    let csv_content = r#"id;size;bedrooms;bathrooms;age;price
1;1200;2;1;5;250000
2;1500;3;2;10;320000
3;2000;4;3;2;450000
4;800;1;1;15;180000
5;2500;5;4;1;650000
6;1800;3;2;8;380000
7;1300;2;2;12;280000"#;
    
    Ok(csv_content.to_string())
}