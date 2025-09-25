use nalgebra::DVector;
use futures::future::join_all;

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU(f64),
    ELU(f64),
    Linear,
    Swish { beta: f64 },
    GELU,
    Mish,
}

impl ActivationFunction {
    /// Apply activation function to input vector
    pub fn apply(&self, input: &DVector<f64>) -> DVector<f64> {
        match self {
            ActivationFunction::Sigmoid => input.map(|x| sigmoid(x)),
            ActivationFunction::ReLU => input.map(|x| x.max(0.0)),
            ActivationFunction::Tanh => input.map(|x| x.tanh()),
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x >= 0.0 { x } else { alpha * x })
            },
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) })
            },
            ActivationFunction::Linear => input.clone(),
            ActivationFunction::Swish { beta } => {
                input.map(|x| x * sigmoid(beta * x))
            },
            ActivationFunction::GELU => {
                input.map(|x| {
                    0.5 * x * (1.0 + (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt() 
                        * (x + 0.044715 * x.powi(3)).tanh())
                })
            },
            ActivationFunction::Mish => {
                input.map(|x| x * (x.exp().ln_1p()).tanh()) // softplus(x) = ln(1 + exp(x))
            },
        }
    }

    /// Apply derivative of activation function to input vector
    pub fn apply_derivative(&self, input: &DVector<f64>) -> DVector<f64> {
        match self {
            ActivationFunction::Sigmoid => {
                input.map(|x| {
                    let s = sigmoid(x);
                    s * (1.0 - s)
                })
            },
            ActivationFunction::ReLU => input.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Tanh => input.map(|x| 1.0 - x.tanh().powi(2)),
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x >= 0.0 { 1.0 } else { *alpha })
            },
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x >= 0.0 { 1.0 } else { alpha * x.exp() })
            },
            ActivationFunction::Linear => DVector::from_element(input.len(), 1.0),
            ActivationFunction::Swish { beta } => {
                input.map(|x| {
                    let s = sigmoid(beta * x);
                    s + beta * x * s * (1.0 - s)
                })
            },
            ActivationFunction::GELU => {
                input.map(|x| {
                    let tanh_arg = (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt() 
                        * (x + 0.044715 * x.powi(3));
                    let tanh_val = tanh_arg.tanh();
                    let sech_sq = 1.0 - tanh_val.powi(2);
                    
                    0.5 * (1.0 + tanh_val) + 0.5 * x * sech_sq * 
                    (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt() * 
                    (1.0 + 3.0 * 0.044715 * x.powi(2))
                })
            },
            ActivationFunction::Mish => {
                input.map(|x| {
                    let softplus = x.exp().ln_1p();
                    let tanh_softplus = softplus.tanh();
                    let sigmoid_x = sigmoid(x);
                    
                    tanh_softplus + x * sigmoid_x * (1.0 - tanh_softplus.powi(2))
                })
            },
        }
    }

    pub async fn apply_async(&self, input: &DVector<f64>) -> DVector<f64> {
        if input.len() > 1000 {
            let result = self.apply(input);
            tokio::task::yield_now().await;
            result
        } else {
            self.apply(input)
        }
    }

    pub async fn apply_batch_async(&self, inputs: &[DVector<f64>]) -> Vec<DVector<f64>> {
        if inputs.len() > 10 {
            let futures: Vec<_> = inputs.iter()
                .map(|input| self.apply_async(input))
                .collect();
            join_all(futures).await
        } else {
            inputs.iter().map(|input| self.apply(input)).collect()
        }
    }

    pub async fn apply_derivative_async(&self, input: &DVector<f64>) -> DVector<f64> {
        if input.len() > 1000 {
            let result = self.apply_derivative(input);
            tokio::task::yield_now().await;
            result
        } else {
            self.apply_derivative(input)
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
