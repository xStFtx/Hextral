use crate::*;
use nalgebra::DVector;

#[test]
fn test_network_creation() {
    let nn = Hextral::new(
        2,
        &[3, 2],
        1,
        ActivationFunction::ReLU,
        Optimizer::Adam { learning_rate: 0.001 },
    );
    
    assert_eq!(nn.architecture(), vec![2, 3, 2, 1]);
    assert_eq!(nn.parameter_count(), 2*3 + 3 + 3*2 + 2 + 2*1 + 1); // weights + biases
}

#[test]
fn test_forward_pass() {
    let nn = Hextral::new(
        2,
        &[3],
        1,
        ActivationFunction::ReLU,
        Optimizer::default(),
    );

    let input = DVector::from_vec(vec![1.0, 2.0]);
    let result = nn.predict(&input);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_training() {
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
    let targets = vec![
        DVector::from_vec(vec![0.0]),
        DVector::from_vec(vec![1.0]),
    ];

    let loss_history = nn.train(&inputs, &targets, 0.01, 5);
    assert_eq!(loss_history.len(), 5);
}

#[test]
fn test_xor_learning() {
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

    let initial_loss = nn.evaluate(&inputs, &targets);
    nn.train(&inputs, &targets, 0.1, 50);
    let final_loss = nn.evaluate(&inputs, &targets);
    
    // Network should learn and reduce loss
    assert!(final_loss < initial_loss);
}

#[test]
fn test_batch_normalization() {
    let mut nn = Hextral::new(
        2,
        &[4, 4],
        1,
        ActivationFunction::ReLU,
        Optimizer::Adam { learning_rate: 0.001 },
    );

    // Enable batch normalization
    nn.enable_batch_norm();
    
    // Set training mode
    nn.set_training_mode(true);

    let inputs = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![
        DVector::from_vec(vec![0.0]),
        DVector::from_vec(vec![1.0]),
    ];

    // Test training with batch norm
    let loss_history = nn.train(&inputs, &targets, 0.01, 5);
    assert_eq!(loss_history.len(), 5);
    
    // Test inference mode
    nn.set_training_mode(false);
    let prediction = nn.predict(&inputs[0]);
    assert_eq!(prediction.len(), 1);
    
    // Test disabling batch norm
    nn.disable_batch_norm();
    let prediction_no_bn = nn.predict(&inputs[0]);
    assert_eq!(prediction_no_bn.len(), 1);
}

#[test]
fn test_new_activation_functions() {
    let input = DVector::from_vec(vec![1.0, -1.0, 0.0, 2.0]);
    
    // Test Swish
    let mut nn_swish = Hextral::new(
        4,
        &[3],
        1,
        ActivationFunction::Swish { beta: 1.0 },
        Optimizer::default(),
    );
    let output_swish = nn_swish.predict(&input);
    assert_eq!(output_swish.len(), 1);
    
    // Test GELU
    let mut nn_gelu = Hextral::new(
        4,
        &[3],
        1,
        ActivationFunction::GELU,
        Optimizer::default(),
    );
    let output_gelu = nn_gelu.predict(&input);
    assert_eq!(output_gelu.len(), 1);
    
    // Test Mish
    let mut nn_mish = Hextral::new(
        4,
        &[3],
        1,
        ActivationFunction::Mish,
        Optimizer::default(),
    );
    let output_mish = nn_mish.predict(&input);
    assert_eq!(output_mish.len(), 1);
    
    // Test that they can be trained
    let targets = vec![DVector::from_vec(vec![0.5])];
    let inputs = vec![input];
    
    let _loss = nn_swish.train(&inputs, &targets, 0.01, 1);
    let _loss = nn_gelu.train(&inputs, &targets, 0.01, 1);
    let _loss = nn_mish.train(&inputs, &targets, 0.01, 1);
}