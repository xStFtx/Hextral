use nalgebra::DVector;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ActivationFunction {
    Sigmoid,
    #[default]
    ReLU,
    Tanh,
    LeakyReLU(f64),
    ELU(f64),
    Linear,
    Swish {
        beta: f64,
    },
    GELU,
    Mish,
    Quaternion,
}

impl ActivationFunction {
    /// Apply activation function to input vector
    pub fn apply(&self, input: &DVector<f64>) -> DVector<f64> {
        match self {
            ActivationFunction::Sigmoid => input.map(sigmoid),
            ActivationFunction::ReLU => input.map(|x| x.max(0.0)),
            ActivationFunction::Tanh => input.map(|x| x.tanh()),
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x >= 0.0 { x } else { alpha * x })
            }
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }
            ActivationFunction::Linear => input.clone(),
            ActivationFunction::Swish { beta } => input.map(|x| x * sigmoid(beta * x)),
            ActivationFunction::GELU => input.map(|x| {
                0.5 * x
                    * (1.0
                        + (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt()
                            * (x + 0.044715 * x.powi(3)).tanh())
            }),
            ActivationFunction::Mish => input.map(|x| x * (x.exp().ln_1p()).tanh()),
            ActivationFunction::Quaternion => quaternion_activation(input),
        }
    }

    /// Apply derivative of activation function to input vector
    pub fn apply_derivative(&self, input: &DVector<f64>) -> DVector<f64> {
        match self {
            ActivationFunction::Sigmoid => input.map(|x| {
                let s = sigmoid(x);
                s * (1.0 - s)
            }),
            ActivationFunction::ReLU => input.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Tanh => input.map(|x| 1.0 - x.tanh().powi(2)),
            ActivationFunction::LeakyReLU(alpha) => {
                input.map(|x| if x >= 0.0 { 1.0 } else { *alpha })
            }
            ActivationFunction::ELU(alpha) => {
                input.map(|x| if x >= 0.0 { 1.0 } else { alpha * x.exp() })
            }
            ActivationFunction::Linear => DVector::from_element(input.len(), 1.0),
            ActivationFunction::Swish { beta } => input.map(|x| {
                let s = sigmoid(beta * x);
                s + beta * x * s * (1.0 - s)
            }),
            ActivationFunction::GELU => input.map(|x| {
                let tanh_arg = (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt()
                    * (x + 0.044715 * x.powi(3));
                let tanh_val = tanh_arg.tanh();
                let sech_sq = 1.0 - tanh_val.powi(2);

                0.5 * (1.0 + tanh_val)
                    + 0.5
                        * x
                        * sech_sq
                        * (std::f64::consts::SQRT_2 / std::f64::consts::PI).sqrt()
                        * (1.0 + 3.0 * 0.044715 * x.powi(2))
            }),
            ActivationFunction::Mish => input.map(|x| {
                let softplus = x.exp().ln_1p();
                let tanh_softplus = softplus.tanh();
                let sigmoid_x = sigmoid(x);

                tanh_softplus + x * sigmoid_x * (1.0 - tanh_softplus.powi(2))
            }),
            ActivationFunction::Quaternion => quaternion_activation_derivative(input),
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn quaternion_activation(input: &DVector<f64>) -> DVector<f64> {
    let len = input.len();
    if len < 4 {
        return input.clone();
    }

    let mut result = input.clone();
    let chunk_size = len / 4;

    for i in 0..chunk_size {
        let base_idx = i * 4;
        if base_idx + 3 < len {
            let w = input[base_idx];
            let x = input[base_idx + 1];
            let y = input[base_idx + 2];
            let z = input[base_idx + 3];

            let norm = (w * w + x * x + y * y + z * z).sqrt();
            if norm > 0.0 {
                result[base_idx] = w / norm;
                result[base_idx + 1] = x / norm;
                result[base_idx + 2] = y / norm;
                result[base_idx + 3] = z / norm;
            }
        }
    }

    result
}

pub fn quaternion_activation_derivative(input: &DVector<f64>) -> DVector<f64> {
    let len = input.len();
    if len < 4 {
        return DVector::from_element(len, 1.0);
    }

    let mut result = DVector::from_element(len, 1.0);
    let chunk_size = len / 4;

    for i in 0..chunk_size {
        let base_idx = i * 4;
        if base_idx + 3 < len {
            let w = input[base_idx];
            let x = input[base_idx + 1];
            let y = input[base_idx + 2];
            let z = input[base_idx + 3];

            let norm_sq = w * w + x * x + y * y + z * z;
            let norm = norm_sq.sqrt();

            if norm > 0.0 {
                let inv_norm = 1.0 / norm;
                let inv_norm_cubed = inv_norm * inv_norm * inv_norm;

                result[base_idx] = inv_norm - w * w * inv_norm_cubed;
                result[base_idx + 1] = inv_norm - x * x * inv_norm_cubed;
                result[base_idx + 2] = inv_norm - y * y * inv_norm_cubed;
                result[base_idx + 3] = inv_norm - z * z * inv_norm_cubed;
            }
        }
    }

    result
}
