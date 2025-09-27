#[cfg(feature = "config")]
use hextral::{ConfigFormat, HextralConfig};

#[cfg(feature = "config")]
use nalgebra::DVector;

#[cfg(feature = "config")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_toml = r#"
        [network]
        input_size = 4
        hidden_layers = [8, 4]
        output_size = 2
        activation = "ReLU"
        optimizer = { Adam = { learning_rate = 0.005, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } }
        loss_function = "MeanSquaredError"
        regularization = "None"
        batch_norm = { enabled = true }

        [training]
        learning_rate = 0.005
        epochs = 10
        batch_size = 16
        [training.early_stopping]
        patience = 3
        min_delta = 0.0001
        restore_best_weights = true
    "#;

    let config = HextralConfig::from_str(config_toml, ConfigFormat::Toml)?;
    let build = config.builder().build()?;
    let model = build.model;

    // Demo prediction with random input vector
    let input = DVector::from_vec(vec![0.25, 0.1, -0.05, 0.9]);
    let prediction = model.predict(&input).await;

    println!("Prediction vector: {:?}", prediction);

    if let Some(training) = build.training {
        println!(
            "Training learning rate from config: {}",
            training.learning_rate
        );
    }

    Ok(())
}

#[cfg(not(feature = "config"))]
fn main() {
    eprintln!("Enable the `config` feature to run this example: cargo run --example config_builder_demo --features config");
}
