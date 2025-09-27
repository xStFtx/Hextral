#![cfg(feature = "config")]

use hextral::{ConfigFormat, HextralConfig};

#[test]
fn builds_network_from_toml_configuration() {
    let config_toml = r#"
        [network]
        input_size = 4
        hidden_layers = [8, 3]
        output_size = 1
    activation = "ReLU"
        optimizer = { Adam = { learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } }
    loss_function = "MeanSquaredError"
    regularization = "None"
        batch_norm = { enabled = true }

        [training]
        learning_rate = 0.01
        epochs = 25
        batch_size = 32
        [training.early_stopping]
        patience = 5
        min_delta = 0.0005
        restore_best_weights = true

        [training.checkpoint]
        filepath = "checkpoints/best_model.bin"
        save_best = true
        save_every = 10
        monitor_loss = true
    "#;

    let config = HextralConfig::from_str(config_toml, ConfigFormat::Toml)
        .expect("configuration should parse");

    let build = config.builder().build().expect("builder should succeed");
    let model = build.model;

    assert_eq!(model.architecture(), vec![4, 8, 3, 1]);
    assert!(model.parameter_count() > 0);

    let training = build.training.expect("training config present");
    assert_eq!(training.learning_rate, 0.01);
    assert_eq!(training.batch_size, Some(32));

    let early = training
        .early_stopping()
        .expect("early stopping config present");
    assert_eq!(early.patience, 5);

    let checkpoint = training.checkpoint().expect("checkpoint config present");
    assert_eq!(checkpoint.filepath, "checkpoints/best_model.bin");
}
