use hextral::{
    Hextral, ActivationFunction, Optimizer, HextralResult,
    dataset::{
        image::{ImageLoader, LabelStrategy, AugmentationConfig},
    }
};
use image::{RgbImage, ImageBuffer};
use nalgebra::DVector;
use std::path::PathBuf;
use tokio::fs;

#[tokio::main]
async fn main() -> HextralResult<()> {
    println!("Hextral Image Dataset Loading Example");
    println!("=====================================");

    // Create sample images
    let temp_dir = create_sample_image_dataset().await?;
    
    // Example 1: Basic image loading
    println!("\nExample 1: Loading images from directory structure");
    
    let image_loader = ImageLoader::new()
    .with_target_size(32, 32)
    .with_normalization(true)
    .with_grayscale(false)
    .with_label_strategy(LabelStrategy::FromDirectory);
    
    let dataset = image_loader.from_directory(&temp_dir).await?;
    
    println!("Loaded image dataset:");
    println!("   Images: {} samples", dataset.metadata.sample_count);
    println!("   Pixels per image: {} (32×32×3)", dataset.metadata.feature_count);
    
    if let Some(target_count) = dataset.metadata.target_count {
        println!("   Classes: {}", target_count);
    }
    
    if let Some(class_names) = &dataset.target_names {
        println!("   Class names: {:?}", class_names);
    }
    
    // Example 2: Grayscale image processing
    println!("\nExample 2: Grayscale image processing");
    
    let grayscale_loader = ImageLoader::new()
    .with_target_size(28, 28)
    .with_grayscale(true)
        .with_normalization(true)
        .with_label_strategy(LabelStrategy::FromDirectory);
    
    let grayscale_dataset = grayscale_loader.from_directory(&temp_dir).await?;
    
    println!("Loaded grayscale dataset:");
    println!("   Images: {} samples", grayscale_dataset.metadata.sample_count);
    println!("   Pixels per image: {} (28×28×1)", grayscale_dataset.metadata.feature_count);
    
    // Example 3: Image augmentation
    println!("\nExample 3: Data augmentation");
    
    let augmentation = AugmentationConfig::new()
    .with_horizontal_flip(0.5)
    .with_rotation(15.0)
    .with_brightness(0.8, 1.2)
    .with_contrast(0.8, 1.2);
    
    let _augmented_loader = ImageLoader::new()
        .with_target_size(64, 64)
        .with_augmentation(augmentation);
    
    println!("Configured augmentation:");
    println!("   Horizontal flip: 50% chance");
    println!("   Rotation: ±15°");
    println!("   Brightness: ±20%");
    println!("   Contrast: ±20%");
    
    // Example 4: Training a simple CNN
    println!("\nExample 4: Training image classifier");
    
    // Use original dataset for training
    let train_dataset = dataset.clone();
    
    // Split data
    let (train_set, test_set) = train_dataset.train_test_split(0.8);
    println!("Split dataset: {} train, {} test", 
             train_set.metadata.sample_count, 
             test_set.metadata.sample_count);
    
    // Convert targets to one-hot encoding
    let num_classes = train_set.target_names.as_ref().map(|names| names.len()).unwrap_or(2);
    
    // Create neural network
    let mut nn = Hextral::new(
    train_set.metadata.feature_count,
    &[512, 256, 128],
        num_classes, // Use the correct number of classes (2)
        ActivationFunction::ReLU,
        Optimizer::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
    );
    
    // Prepare training data
    let train_inputs = &train_set.features;
    let train_targets = train_set.targets.as_ref().unwrap();
    
    let one_hot_targets: Vec<DVector<f64>> = train_targets.iter()
        .map(|target| {
            let class_id = target[0] as usize;
            let mut one_hot = vec![0.0; num_classes];
            if class_id < num_classes {
                one_hot[class_id] = 1.0;
            }
            DVector::from_vec(one_hot)
        })
        .collect();
    
    println!("Training neural network...");
    let result = nn.train(
        train_inputs,
        &one_hot_targets,
        0.01,
        10,      // Fewer epochs for demo
        Some(8), // Small batch size
        None,    // No validation data
        None,    // No validation targets  
        None,    // No early stopping
        None,    // No checkpoints
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
    
    // Example 5: Evaluation and prediction
    println!("\nExample 5: Model evaluation");
    
    let test_inputs = &test_set.features;
    let test_targets = test_set.targets.as_ref().unwrap();
    
    let mut correct_predictions = 0;
    let total_predictions = test_inputs.len();
    
    for (i, (input, target)) in test_inputs.iter().zip(test_targets.iter()).enumerate() {
        let prediction = nn.predict(input).await;
        
        // Find predicted class
        let predicted_class = prediction.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let actual_class = target[0] as usize;
        
        if predicted_class == actual_class {
            correct_predictions += 1;
        }
        
        // Show first few predictions
        if i < 3 {
            println!("   Sample {}: predicted {}, actual {} (confidence: {:.2})", 
                     i + 1, predicted_class, actual_class, 
                     prediction[predicted_class]);
        }
    }
    
    let accuracy = correct_predictions as f64 / total_predictions as f64;
    println!("Test accuracy: {:.1}% ({}/{} correct)", 
             accuracy * 100.0, correct_predictions, total_predictions);
    
    // Example 6: Single image prediction
    println!("\nExample 6: Single image prediction");
    
    let single_loader = ImageLoader::new()
        .with_target_size(32, 32)
        .with_normalization(true)
        .with_label_strategy(LabelStrategy::None);
    
    // Create a sample image path
    let sample_image_path = temp_dir.join("cats").join("cat_0.png");
    
    if sample_image_path.exists() {
        let image_vector = single_loader.load_image(&sample_image_path).await?;
        let prediction = nn.predict(&image_vector).await;
        
        let predicted_class = prediction.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        if let Some(class_names) = &dataset.target_names {
            if predicted_class < class_names.len() {
                println!("Predicted class for {}: {} (confidence: {:.2})", 
                         sample_image_path.display(),
                         class_names[predicted_class],
                         prediction[predicted_class]);
            }
        }
    }
    
    // Cleanup
    fs::remove_dir_all(&temp_dir).await?;
    
    println!("\nImage loading examples completed successfully!");
    Ok(())
}

/// Create sample image dataset with directory structure
async fn create_sample_image_dataset() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir().join("hextral_image_demo");
    
    // Create directory structure
    let cats_dir = temp_dir.join("cats");
    let dogs_dir = temp_dir.join("dogs");
    
    fs::create_dir_all(&cats_dir).await?;
    fs::create_dir_all(&dogs_dir).await?;
    
    // Create sample cat images (simple colored rectangles)
    for i in 0..5 {
        let img = create_sample_image(255, 200, 100); // Orange-ish for cats
        let path = cats_dir.join(format!("cat_{}.png", i));
        img.save(&path)?;
    }
    
    // Create sample dog images (different colored rectangles) 
    for i in 0..5 {
        let img = create_sample_image(150, 100, 200); // Purple-ish for dogs
        let path = dogs_dir.join(format!("dog_{}.png", i));
        img.save(&path)?;
    }
    
    println!("Created sample dataset at: {}", temp_dir.display());
    println!("   Cats: 5 images");
    println!("   Dogs: 5 images");
    
    Ok(temp_dir)
}

/// Create a simple colored image for demonstration
fn create_sample_image(r: u8, g: u8, b: u8) -> RgbImage {
    let width = 64;
    let height = 64;
    
    ImageBuffer::from_fn(width, height, |x, y| {
        // Create some simple patterns to make images distinct
        let pattern_r = ((r as f32 * (1.0 + 0.2 * (x as f32 / width as f32).sin())) as u8).min(255);
        let pattern_g = ((g as f32 * (1.0 + 0.2 * (y as f32 / height as f32).sin())) as u8).min(255);
        let pattern_b = ((b as f32 * (1.0 + 0.1 * ((x + y) as f32 / (width + height) as f32).sin())) as u8).min(255);
        
        image::Rgb([pattern_r, pattern_g, pattern_b])
    })
}