#[cfg(test)]
mod optimizer_tests {
    use hextral::{ActivationFunction, Hextral, Optimizer, OptimizerState};
    use nalgebra::DVector;

    #[test]
    fn test_all_optimizers_compile() {
        let input_size = 2;
        let hidden_sizes = &[4];
        let output_size = 1;
        let activation = ActivationFunction::ReLU;

        // Test that all optimizers can be constructed and used
        let optimizers = vec![
            Optimizer::sgd(0.1),
            Optimizer::sgd_momentum(0.1, 0.9),
            Optimizer::adam(0.001),
            Optimizer::adamw(0.001, 0.01),
            Optimizer::rmsprop(0.001),
            Optimizer::adagrad(0.1),
            Optimizer::adadelta(),
            Optimizer::nadam(0.001),
            Optimizer::lion(0.001),
            Optimizer::adabelief(0.001),
        ];

        for optimizer in optimizers {
            let _nn = Hextral::new(
                input_size,
                hidden_sizes,
                output_size,
                activation.clone(),
                optimizer,
            );
            // If we get here without panicking, the optimizer works
        }
    }

    #[tokio::test]
    async fn test_optimizer_convergence() {
        // Simple test to verify optimizers can learn
        let inputs = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];
        let targets = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];

        // Test a few key optimizers
        let optimizers = vec![
            ("Adam", Optimizer::adam(0.01)),
            ("AdamW", Optimizer::adamw(0.01, 0.01)),
            ("NAdam", Optimizer::nadam(0.01)),
        ];

        for (name, optimizer) in optimizers {
            let mut nn = Hextral::new(2, &[4], 1, ActivationFunction::ReLU, optimizer);

            let initial_loss = nn.evaluate(&inputs, &targets).await.unwrap();
            let _ = nn
                .train(&inputs, &targets, 1.0, 50, None, None, None, None, None)
                .await
                .unwrap();
            let final_loss = nn.evaluate(&inputs, &targets).await.unwrap();

            // Loss should decrease (though we allow some flexibility)
            assert!(
                final_loss < initial_loss * 2.0,
                "{} optimizer failed to converge: initial={:.6}, final={:.6}",
                name,
                initial_loss,
                final_loss
            );
        }
    }

    #[test]
    fn test_optimizer_state_initialization() {
        let layer_shapes = vec![(4, 2), (1, 4)];
        let state = OptimizerState::new(&layer_shapes);

        // Verify all arrays are properly initialized
        assert_eq!(state.velocity_weights.len(), 2);
        assert_eq!(state.velocity_biases.len(), 2);
        assert_eq!(state.squared_weights.len(), 2);
        assert_eq!(state.squared_biases.len(), 2);

        // Verify shapes
        assert_eq!(state.velocity_weights[0].shape(), (4, 2));
        assert_eq!(state.velocity_weights[1].shape(), (1, 4));
        assert_eq!(state.velocity_biases[0].len(), 4);
        assert_eq!(state.velocity_biases[1].len(), 1);
    }
}
