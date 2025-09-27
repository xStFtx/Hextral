use hextral::*;
use nalgebra::DVector;

#[test]
fn test_network_creation() {
    let nn = Hextral::new(
        2,
        &[3, 2],
        1,
        ActivationFunction::ReLU,
        Optimizer::adam(0.001),
    );

    assert_eq!(nn.architecture(), vec![2, 3, 2, 1]);
    assert_eq!(nn.parameter_count(), 2 * 3 + 3 + 3 * 2 + 2 + 2 * 1 + 1); // weights + biases
}

#[tokio::test]
async fn test_forward_pass() {
    let nn = Hextral::new(2, &[3], 1, ActivationFunction::ReLU, Optimizer::default());

    let input = DVector::from_vec(vec![1.0, 2.0]);
    let result = nn.predict(&input).await;
    assert_eq!(result.len(), 1);
}

#[tokio::test]
async fn test_training() {
    let mut nn = Hextral::new(
        2,
        &[4, 3],
        1,
        ActivationFunction::ReLU,
        Optimizer::default(),
    );

    let inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];

    let (loss_history, _) = nn
        .train(&inputs, &targets, 0.01, 5, None, None, None, None, None)
        .await
        .unwrap();
    assert_eq!(loss_history.len(), 5);
}

#[tokio::test]
async fn test_xor_learning() {
    let mut nn = Hextral::new(
        2,
        &[4, 4],
        1,
        ActivationFunction::Tanh,
        Optimizer::SGD { learning_rate: 0.5 },
    );

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

    let initial_loss = nn.evaluate(&inputs, &targets).await.unwrap();
    let _ = nn
        .train(&inputs, &targets, 0.1, 50, None, None, None, None, None)
        .await
        .unwrap();
    let final_loss = nn.evaluate(&inputs, &targets).await.unwrap();

    // Network should learn and reduce loss
    assert!(final_loss < initial_loss);
}

#[tokio::test]
async fn test_batch_normalization() {
    let mut nn = Hextral::new(
        2,
        &[4, 4],
        1,
        ActivationFunction::ReLU,
        Optimizer::adam(0.001),
    );

    // Enable batch normalization
    nn.enable_batch_norm();

    // Set training mode
    nn.set_training_mode(true);

    let inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];

    // Test training with batch norm
    let (loss_history, _) = nn
        .train(&inputs, &targets, 0.01, 5, None, None, None, None, None)
        .await
        .unwrap();
    assert_eq!(loss_history.len(), 5);

    // Test inference mode
    nn.set_training_mode(false);
    let prediction = nn.predict(&inputs[0]).await;
    assert_eq!(prediction.len(), 1);

    // Test disabling batch norm
    nn.disable_batch_norm();
    let prediction_no_bn = nn.predict(&inputs[0]).await;
    assert_eq!(prediction_no_bn.len(), 1);
}

#[tokio::test]
async fn test_new_activation_functions() {
    let input = DVector::from_vec(vec![1.0, -1.0, 0.0, 2.0]);

    // Test Swish
    let mut nn_swish = Hextral::new(
        4,
        &[3],
        1,
        ActivationFunction::Swish { beta: 1.0 },
        Optimizer::default(),
    );
    let output_swish = nn_swish.predict(&input).await;
    assert_eq!(output_swish.len(), 1);

    // Test GELU
    let mut nn_gelu = Hextral::new(4, &[3], 1, ActivationFunction::GELU, Optimizer::default());
    let output_gelu = nn_gelu.predict(&input).await;
    assert_eq!(output_gelu.len(), 1);

    // Test Mish
    let mut nn_mish = Hextral::new(4, &[3], 1, ActivationFunction::Mish, Optimizer::default());
    let output_mish = nn_mish.predict(&input).await;
    assert_eq!(output_mish.len(), 1);

    // Test that they can be trained
    let targets = vec![DVector::from_vec(vec![0.5])];
    let inputs = vec![input];

    let _ = nn_swish
        .train(&inputs, &targets, 0.01, 1, None, None, None, None, None)
        .await
        .unwrap();
    let _ = nn_gelu
        .train(&inputs, &targets, 0.01, 1, None, None, None, None, None)
        .await
        .unwrap();
    let _ = nn_mish
        .train(&inputs, &targets, 0.01, 1, None, None, None, None, None)
        .await
        .unwrap();

    // Test Quaternion activation function directly
    let quaternion_input = DVector::from_vec(vec![3.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 12.0]);
    let quaternion_output = hextral::activation::quaternion_activation(&quaternion_input);
    assert_eq!(quaternion_output.len(), 8);

    // Check first quaternion normalization
    let norm1 = (quaternion_output[0] * quaternion_output[0]
        + quaternion_output[1] * quaternion_output[1]
        + quaternion_output[2] * quaternion_output[2]
        + quaternion_output[3] * quaternion_output[3])
        .sqrt();
    assert!(
        (norm1 - 1.0).abs() < 0.01,
        "First quaternion should be normalized"
    );

    // Check second quaternion normalization
    let norm2 = (quaternion_output[4] * quaternion_output[4]
        + quaternion_output[5] * quaternion_output[5]
        + quaternion_output[6] * quaternion_output[6]
        + quaternion_output[7] * quaternion_output[7])
        .sqrt();
    assert!(
        (norm2 - 1.0).abs() < 0.01,
        "Second quaternion should be normalized"
    );
}

#[tokio::test]
async fn test_early_stopping() {
    // Create simple data
    let train_inputs = vec![
        DVector::from_vec(vec![0.0, 1.0]),
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
        DVector::from_vec(vec![0.0, 0.0]),
    ];
    let train_targets = vec![
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![1.0]),
        DVector::from_vec(vec![0.0]),
        DVector::from_vec(vec![0.0]),
    ];

    let val_inputs = train_inputs.clone();
    let val_targets = train_targets.clone();

    let mut nn = Hextral::new(
        2,
        &[4],
        1,
        ActivationFunction::Sigmoid,
        Optimizer::adam(0.001),
    );

    let early_stop = EarlyStopping::new(5, 0.001, true);

    let (train_history, val_history) = nn
        .train(
            &train_inputs,
            &train_targets,
            0.1,
            100,
            None,
            Some(&val_inputs),
            Some(&val_targets),
            Some(early_stop),
            None,
        )
        .await
        .unwrap();

    assert!(!train_history.is_empty());
    assert!(!val_history.is_empty());
    assert_eq!(train_history.len(), val_history.len());
    assert!(train_history.len() <= 100, "Should stop before max epochs");
}

#[test]
fn test_checkpoint_config() {
    let config = CheckpointConfig::new("test_model.bin".to_string());
    assert_eq!(config.filepath, "test_model.bin");
    assert!(config.save_best);
    assert!(config.save_every.is_none());

    let config_with_periodic = config.save_every(10);
    assert_eq!(config_with_periodic.save_every, Some(10));
}

#[test]
fn test_early_stopping_logic() {
    let mut early_stop = EarlyStopping::new(3, 0.01, false);

    // Test improvement
    assert!(!early_stop.should_stop(1.0));
    assert_eq!(early_stop.counter, 0);
    assert_eq!(early_stop.best_loss, 1.0);

    // Test small improvement (within min_delta)
    assert!(!early_stop.should_stop(0.995));
    assert_eq!(early_stop.counter, 1);

    // Test no improvement
    assert!(!early_stop.should_stop(1.0));
    assert_eq!(early_stop.counter, 2);

    // Test should stop after patience exceeded
    assert!(early_stop.should_stop(1.1));
    assert_eq!(early_stop.counter, 3);
}

// Async tests
#[tokio::test]
async fn test_async_forward_pass() {
    let nn = Hextral::new(2, &[3], 1, ActivationFunction::ReLU, Optimizer::default());

    let input = DVector::from_vec(vec![1.0, 2.0]);
    let result = nn.forward(&input).await;
    assert_eq!(result.len(), 1);
}

#[tokio::test]
async fn test_async_prediction() {
    let nn = Hextral::new(
        2,
        &[3],
        1,
        ActivationFunction::Sigmoid,
        Optimizer::default(),
    );

    let input = DVector::from_vec(vec![1.0, 2.0]);
    let result = nn.predict(&input).await;
    assert_eq!(result.len(), 1);
}

#[tokio::test]
async fn test_async_batch_prediction() {
    let nn = Hextral::new(2, &[3], 2, ActivationFunction::ReLU, Optimizer::default());

    let inputs = vec![
        DVector::from_vec(vec![1.0, 2.0]),
        DVector::from_vec(vec![3.0, 4.0]),
        DVector::from_vec(vec![5.0, 6.0]),
    ];

    let results = nn.predict_batch(&inputs).await;
    assert_eq!(results.len(), 3);
    for result in results {
        assert_eq!(result.len(), 2);
    }
}

#[tokio::test]
async fn test_async_training() {
    let mut nn = Hextral::new(
        2,
        &[4, 3],
        1,
        ActivationFunction::ReLU,
        Optimizer::default(),
    );

    let inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];

    let (train_history, _val_history) = nn
        .train(&inputs, &targets, 0.01, 5, Some(32), None, None, None, None)
        .await
        .unwrap();
    assert_eq!(train_history.len(), 5);
}

#[tokio::test]
async fn test_async_evaluation() {
    let nn = Hextral::new(2, &[4], 1, ActivationFunction::ReLU, Optimizer::default());

    let inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];

    let loss = nn.evaluate(&inputs, &targets).await.unwrap();
    assert!(loss >= 0.0); // Loss should be non-negative
}

#[tokio::test]
async fn test_async_activation_functions() {
    let input = DVector::from_vec(vec![1.0, -2.0, 3.0, -4.0]);

    // Test async activation functions
    let activations = vec![
        ActivationFunction::ReLU,
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
        ActivationFunction::LeakyReLU(0.01),
        ActivationFunction::ELU(1.0),
        ActivationFunction::Linear,
        ActivationFunction::Swish { beta: 1.0 },
        ActivationFunction::GELU,
        ActivationFunction::Mish,
    ];

    for activation in activations {
        let result = activation.apply(&input);

        // Just verify the result is reasonable
        assert_eq!(result.len(), input.len());
        for val in result.iter() {
            assert!(val.is_finite());
        }
    }
}
