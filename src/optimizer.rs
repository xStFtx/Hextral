use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Optimizer {
    SGD { 
        learning_rate: f64 
    },
    SGDMomentum { 
        learning_rate: f64, 
        momentum: f64 
    },
    Adam { 
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    AdamW {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    },
    RMSprop {
        learning_rate: f64,
        alpha: f64,
        epsilon: f64,
    },
    AdaGrad {
        learning_rate: f64,
        epsilon: f64,
    },
    AdaDelta {
        rho: f64,
        epsilon: f64,
    },
    NAdam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    Lion {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
    },
    AdaBelief {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::Adam { 
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    // For momentum-based optimizers
    pub velocity_weights: Vec<DMatrix<f64>>,
    pub velocity_biases: Vec<DVector<f64>>,
    
    // For Adam-family optimizers (second moment)
    pub squared_weights: Vec<DMatrix<f64>>,
    pub squared_biases: Vec<DVector<f64>>,
    
    // For AdaBelief (centralized second moment)
    pub centralized_weights: Vec<DMatrix<f64>>,
    pub centralized_biases: Vec<DVector<f64>>,
    
    // For AdaGrad/RMSprop (accumulated gradients)
    pub accumulated_weights: Vec<DMatrix<f64>>,
    pub accumulated_biases: Vec<DVector<f64>>,
    
    // For AdaDelta (accumulated deltas)
    pub delta_weights: Vec<DMatrix<f64>>,
    pub delta_biases: Vec<DVector<f64>>,
    
    // Time step for bias correction
    pub time_step: usize,
}

impl OptimizerState {
    pub fn new(layer_shapes: &[(usize, usize)]) -> Self {
        let num_layers = layer_shapes.len();
        
        OptimizerState {
            velocity_weights: (0..num_layers)
                .map(|i| DMatrix::zeros(layer_shapes[i].0, layer_shapes[i].1))
                .collect(),
            velocity_biases: (0..num_layers)
                .map(|i| DVector::zeros(layer_shapes[i].0))
                .collect(),
            squared_weights: (0..num_layers)
                .map(|i| DMatrix::zeros(layer_shapes[i].0, layer_shapes[i].1))
                .collect(),
            squared_biases: (0..num_layers)
                .map(|i| DVector::zeros(layer_shapes[i].0))
                .collect(),
            centralized_weights: (0..num_layers)
                .map(|i| DMatrix::zeros(layer_shapes[i].0, layer_shapes[i].1))
                .collect(),
            centralized_biases: (0..num_layers)
                .map(|i| DVector::zeros(layer_shapes[i].0))
                .collect(),
            accumulated_weights: (0..num_layers)
                .map(|i| DMatrix::zeros(layer_shapes[i].0, layer_shapes[i].1))
                .collect(),
            accumulated_biases: (0..num_layers)
                .map(|i| DVector::zeros(layer_shapes[i].0))
                .collect(),
            delta_weights: (0..num_layers)
                .map(|i| DMatrix::zeros(layer_shapes[i].0, layer_shapes[i].1))
                .collect(),
            delta_biases: (0..num_layers)
                .map(|i| DVector::zeros(layer_shapes[i].0))
                .collect(),
            time_step: 0,
        }
    }
}

impl Optimizer {
    pub fn update_parameters(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        state: &mut OptimizerState,
        layer_idx: usize,
        base_learning_rate: f64,
    ) {
        match self {
            Optimizer::SGD { learning_rate } => {
                self.sgd_update(weights, biases, weight_grad, bias_grad, *learning_rate * base_learning_rate);
            },
            
            Optimizer::SGDMomentum { learning_rate, momentum } => {
                self.sgd_momentum_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *momentum,
                    state, layer_idx
                );
            },
            
            Optimizer::Adam { learning_rate, beta1, beta2, epsilon } => {
                self.adam_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *beta1, *beta2, *epsilon,
                    state, layer_idx
                );
            },
            
            Optimizer::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                self.adamw_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *beta1, *beta2, *epsilon, *weight_decay,
                    state, layer_idx
                );
            },
            
            Optimizer::RMSprop { learning_rate, alpha, epsilon } => {
                self.rmsprop_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *alpha, *epsilon,
                    state, layer_idx
                );
            },
            
            Optimizer::AdaGrad { learning_rate, epsilon } => {
                self.adagrad_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *epsilon,
                    state, layer_idx
                );
            },
            
            Optimizer::AdaDelta { rho, epsilon } => {
                self.adadelta_update(
                    weights, biases, weight_grad, bias_grad,
                    *rho, *epsilon,
                    state, layer_idx
                );
            },
            
            Optimizer::NAdam { learning_rate, beta1, beta2, epsilon } => {
                self.nadam_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *beta1, *beta2, *epsilon,
                    state, layer_idx
                );
            },
            
            Optimizer::Lion { learning_rate, beta1, beta2 } => {
                self.lion_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *beta1, *beta2,
                    state, layer_idx
                );
            },
            
            Optimizer::AdaBelief { learning_rate, beta1, beta2, epsilon } => {
                self.adabelief_update(
                    weights, biases, weight_grad, bias_grad,
                    *learning_rate * base_learning_rate, *beta1, *beta2, *epsilon,
                    state, layer_idx
                );
            },
        }
    }

    fn sgd_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
    ) {
        *weights -= weight_grad * learning_rate;
        *biases -= bias_grad * learning_rate;
    }

    fn sgd_momentum_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        momentum: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        // Update velocity
        state.velocity_weights[layer_idx] = 
            &state.velocity_weights[layer_idx] * momentum + weight_grad * learning_rate;
        state.velocity_biases[layer_idx] = 
            &state.velocity_biases[layer_idx] * momentum + bias_grad * learning_rate;
        
        // Update parameters
        *weights -= &state.velocity_weights[layer_idx];
        *biases -= &state.velocity_biases[layer_idx];
    }

    fn adam_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        state.time_step += 1;
        let t = state.time_step as f64;
        
        // Update biased first moment estimate
        state.velocity_weights[layer_idx] = 
            &state.velocity_weights[layer_idx] * beta1 + weight_grad * (1.0 - beta1);
        state.velocity_biases[layer_idx] = 
            &state.velocity_biases[layer_idx] * beta1 + bias_grad * (1.0 - beta1);
        
        // Update biased second raw moment estimate
        let weight_grad_sq = weight_grad.component_mul(weight_grad);
        let bias_grad_sq = bias_grad.component_mul(bias_grad);
        
        state.squared_weights[layer_idx] = 
            &state.squared_weights[layer_idx] * beta2 + &weight_grad_sq * (1.0 - beta2);
        state.squared_biases[layer_idx] = 
            &state.squared_biases[layer_idx] * beta2 + &bias_grad_sq * (1.0 - beta2);
        
        // Compute bias-corrected first moment estimate
        let m_hat_weights = &state.velocity_weights[layer_idx] / (1.0 - beta1.powf(t));
        let m_hat_biases = &state.velocity_biases[layer_idx] / (1.0 - beta1.powf(t));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat_weights = &state.squared_weights[layer_idx] / (1.0 - beta2.powf(t));
        let v_hat_biases = &state.squared_biases[layer_idx] / (1.0 - beta2.powf(t));
        
        // Update parameters
        let weight_update = m_hat_weights.component_div(&(v_hat_weights.map(|x| x.sqrt() + epsilon)));
        let bias_update = m_hat_biases.component_div(&(v_hat_biases.map(|x| x.sqrt() + epsilon)));
        
        *weights -= &weight_update * learning_rate;
        *biases -= &bias_update * learning_rate;
    }

    fn adamw_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        // Apply weight decay to weights (not biases)
        let weight_grad_with_decay = weight_grad + &*weights * weight_decay;
        
        // Use Adam update with modified weight gradients
        self.adam_update(
            weights, biases, &weight_grad_with_decay, bias_grad,
            learning_rate, beta1, beta2, epsilon, state, layer_idx
        );
    }

    fn rmsprop_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        alpha: f64,
        epsilon: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        // Update moving average of squared gradients
        let weight_grad_sq = weight_grad.component_mul(weight_grad);
        let bias_grad_sq = bias_grad.component_mul(bias_grad);
        
        state.squared_weights[layer_idx] = 
            &state.squared_weights[layer_idx] * alpha + &weight_grad_sq * (1.0 - alpha);
        state.squared_biases[layer_idx] = 
            &state.squared_biases[layer_idx] * alpha + &bias_grad_sq * (1.0 - alpha);
        
        // Update parameters
        let weight_update = weight_grad.component_div(&state.squared_weights[layer_idx].map(|x| x.sqrt() + epsilon));
        let bias_update = bias_grad.component_div(&state.squared_biases[layer_idx].map(|x| x.sqrt() + epsilon));
        
        *weights -= &weight_update * learning_rate;
        *biases -= &bias_update * learning_rate;
    }

    fn adagrad_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        epsilon: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        // Accumulate squared gradients
        let weight_grad_sq = weight_grad.component_mul(weight_grad);
        let bias_grad_sq = bias_grad.component_mul(bias_grad);
        
        state.accumulated_weights[layer_idx] += &weight_grad_sq;
        state.accumulated_biases[layer_idx] += &bias_grad_sq;
        
        // Update parameters
        let weight_update = weight_grad.component_div(&state.accumulated_weights[layer_idx].map(|x| x.sqrt() + epsilon));
        let bias_update = bias_grad.component_div(&state.accumulated_biases[layer_idx].map(|x| x.sqrt() + epsilon));
        
        *weights -= &weight_update * learning_rate;
        *biases -= &bias_update * learning_rate;
    }

    fn adadelta_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        rho: f64,
        epsilon: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        // Update accumulated squared gradients
        let weight_grad_sq = weight_grad.component_mul(weight_grad);
        let bias_grad_sq = bias_grad.component_mul(bias_grad);
        
        state.accumulated_weights[layer_idx] = 
            &state.accumulated_weights[layer_idx] * rho + &weight_grad_sq * (1.0 - rho);
        state.accumulated_biases[layer_idx] = 
            &state.accumulated_biases[layer_idx] * rho + &bias_grad_sq * (1.0 - rho);
        
        // Compute updates
        let rms_grad_weights = state.accumulated_weights[layer_idx].map(|x| (x + epsilon).sqrt());
        let rms_grad_biases = state.accumulated_biases[layer_idx].map(|x| (x + epsilon).sqrt());
        let rms_delta_weights = state.delta_weights[layer_idx].map(|x| (x + epsilon).sqrt());
        let rms_delta_biases = state.delta_biases[layer_idx].map(|x| (x + epsilon).sqrt());
        
        let weight_update = weight_grad.component_mul(&rms_delta_weights).component_div(&rms_grad_weights);
        let bias_update = bias_grad.component_mul(&rms_delta_biases).component_div(&rms_grad_biases);
        
        // Update accumulated squared deltas
        let weight_update_sq = weight_update.component_mul(&weight_update);
        let bias_update_sq = bias_update.component_mul(&bias_update);
        
        state.delta_weights[layer_idx] = 
            &state.delta_weights[layer_idx] * rho + &weight_update_sq * (1.0 - rho);
        state.delta_biases[layer_idx] = 
            &state.delta_biases[layer_idx] * rho + &bias_update_sq * (1.0 - rho);
        
        // Update parameters
        *weights -= &weight_update;
        *biases -= &bias_update;
    }

    fn nadam_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        state.time_step += 1;
        let t = state.time_step as f64;
        
        // Update biased first moment estimate
        state.velocity_weights[layer_idx] = 
            &state.velocity_weights[layer_idx] * beta1 + weight_grad * (1.0 - beta1);
        state.velocity_biases[layer_idx] = 
            &state.velocity_biases[layer_idx] * beta1 + bias_grad * (1.0 - beta1);
        
        // Update biased second raw moment estimate
        let weight_grad_sq = weight_grad.component_mul(weight_grad);
        let bias_grad_sq = bias_grad.component_mul(bias_grad);
        
        state.squared_weights[layer_idx] = 
            &state.squared_weights[layer_idx] * beta2 + &weight_grad_sq * (1.0 - beta2);
        state.squared_biases[layer_idx] = 
            &state.squared_biases[layer_idx] * beta2 + &bias_grad_sq * (1.0 - beta2);
        
        // Bias correction
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);
        
        // Nesterov momentum
        let m_hat_weights = (&state.velocity_weights[layer_idx] * beta1 + weight_grad * (1.0 - beta1)) / bias_correction1;
        let m_hat_biases = (&state.velocity_biases[layer_idx] * beta1 + bias_grad * (1.0 - beta1)) / bias_correction1;
        
        let v_hat_weights = &state.squared_weights[layer_idx] / bias_correction2;
        let v_hat_biases = &state.squared_biases[layer_idx] / bias_correction2;
        
        // Update parameters
        let weight_update = m_hat_weights.component_div(&(v_hat_weights.map(|x| x.sqrt() + epsilon)));
        let bias_update = m_hat_biases.component_div(&(v_hat_biases.map(|x| x.sqrt() + epsilon)));
        
        *weights -= &weight_update * learning_rate;
        *biases -= &bias_update * learning_rate;
    }

    fn lion_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        // Lion optimizer: c_t = β1 * m_t-1 + (1 - β1) * g_t
        let c_weights = &state.velocity_weights[layer_idx] * beta1 + weight_grad * (1.0 - beta1);
        let c_biases = &state.velocity_biases[layer_idx] * beta1 + bias_grad * (1.0 - beta1);
        
        // Update parameters: θ_t = θ_t-1 - η * sign(c_t)
        let weight_update = c_weights.map(|x| x.signum());
        let bias_update = c_biases.map(|x| x.signum());
        
        *weights -= &weight_update * learning_rate;
        *biases -= &bias_update * learning_rate;
        
        // Update momentum: m_t = β2 * m_t-1 + (1 - β2) * g_t
        state.velocity_weights[layer_idx] = 
            &state.velocity_weights[layer_idx] * beta2 + weight_grad * (1.0 - beta2);
        state.velocity_biases[layer_idx] = 
            &state.velocity_biases[layer_idx] * beta2 + bias_grad * (1.0 - beta2);
    }

    fn adabelief_update(
        &self,
        weights: &mut DMatrix<f64>,
        biases: &mut DVector<f64>,
        weight_grad: &DMatrix<f64>,
        bias_grad: &DVector<f64>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        state: &mut OptimizerState,
        layer_idx: usize,
    ) {
        state.time_step += 1;
        let t = state.time_step as f64;
        
        // Update biased first moment estimate
        state.velocity_weights[layer_idx] = 
            &state.velocity_weights[layer_idx] * beta1 + weight_grad * (1.0 - beta1);
        state.velocity_biases[layer_idx] = 
            &state.velocity_biases[layer_idx] * beta1 + bias_grad * (1.0 - beta1);
        
        // Centralize gradients (subtract mean)
        let centered_weight_grad = weight_grad - &state.velocity_weights[layer_idx];
        let centered_bias_grad = bias_grad - &state.velocity_biases[layer_idx];
        
        // Update biased second moment of the centered gradients
        let centered_weight_sq = centered_weight_grad.component_mul(&centered_weight_grad);
        let centered_bias_sq = centered_bias_grad.component_mul(&centered_bias_grad);
        
        state.centralized_weights[layer_idx] = 
            &state.centralized_weights[layer_idx] * beta2 + &centered_weight_sq * (1.0 - beta2);
        state.centralized_biases[layer_idx] = 
            &state.centralized_biases[layer_idx] * beta2 + &centered_bias_sq * (1.0 - beta2);
        
        // Bias correction
        let m_hat_weights = &state.velocity_weights[layer_idx] / (1.0 - beta1.powf(t));
        let m_hat_biases = &state.velocity_biases[layer_idx] / (1.0 - beta1.powf(t));
        
        let s_hat_weights = &state.centralized_weights[layer_idx] / (1.0 - beta2.powf(t));
        let s_hat_biases = &state.centralized_biases[layer_idx] / (1.0 - beta2.powf(t));
        
        // Update parameters
        let weight_update = m_hat_weights.component_div(&(s_hat_weights.map(|x| x.sqrt() + epsilon)));
        let bias_update = m_hat_biases.component_div(&(s_hat_biases.map(|x| x.sqrt() + epsilon)));
        
        *weights -= &weight_update * learning_rate;
        *biases -= &bias_update * learning_rate;
    }
}

// Convenience constructors for common optimizer configurations
impl Optimizer {
    pub fn sgd(learning_rate: f64) -> Self {
        Optimizer::SGD { learning_rate }
    }
    
    pub fn sgd_momentum(learning_rate: f64, momentum: f64) -> Self {
        Optimizer::SGDMomentum { learning_rate, momentum }
    }
    
    pub fn adam(learning_rate: f64) -> Self {
        Optimizer::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
    
    pub fn adamw(learning_rate: f64, weight_decay: f64) -> Self {
        Optimizer::AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
        }
    }
    
    pub fn rmsprop(learning_rate: f64) -> Self {
        Optimizer::RMSprop {
            learning_rate,
            alpha: 0.99,
            epsilon: 1e-8,
        }
    }
    
    pub fn adagrad(learning_rate: f64) -> Self {
        Optimizer::AdaGrad {
            learning_rate,
            epsilon: 1e-10,
        }
    }
    
    pub fn adadelta() -> Self {
        Optimizer::AdaDelta {
            rho: 0.95,
            epsilon: 1e-6,
        }
    }
    
    pub fn nadam(learning_rate: f64) -> Self {
        Optimizer::NAdam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
    
    pub fn lion(learning_rate: f64) -> Self {
        Optimizer::Lion {
            learning_rate,
            beta1: 0.9,
            beta2: 0.99,
        }
    }
    
    pub fn adabelief(learning_rate: f64) -> Self {
        Optimizer::AdaBelief {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
        }
    }
}
