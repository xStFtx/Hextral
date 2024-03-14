use nalgebra::{DVector, DMatrix};

enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
}

enum Regularization {
    L2(f64),
    L1(f64),
}

struct Hextral {
    h: DMatrix<f64>,
    qft: f64,
    laplace: f64,
}

impl Hextral {
    fn new(qft: f64, laplace: f64) -> Self {
        let h = DMatrix::from_fn(10, 10, |_, _| rand::random::<f64>() * 0.1);

        Hextral { h, qft, laplace }
    }

    fn forward_pass(&self, input: &DVector<f64>, activation: ActivationFunction) -> DVector<f64> {
        let output = &self.h * input;

        let output = match activation {
            ActivationFunction::Sigmoid => output.map(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::ReLU => output.map(|x| x.max(0.0)),
            ActivationFunction::Tanh => output.map(|x| x.tanh()),
        };

        output
    }

    fn train(&mut self, inputs: &[DVector<f64>], targets: &[DVector<f64>], learning_rate: f64, regularization: Regularization, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = self.forward_pass(input, ActivationFunction::Sigmoid);
                let loss_gradient = output - target;
                let gradients = &loss_gradient * input.transpose();

                self.update_parameters(learning_rate, &gradients, &regularization); // Pass regularization as reference
            }
        }
    }

    fn predict(&self, input: &DVector<f64>) -> DVector<f64> {
        self.forward_pass(input, ActivationFunction::Sigmoid)
    }

    fn evaluate(&self, inputs: &[DVector<f64>], targets: &[DVector<f64>]) -> f64 {
        let mut total_loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.predict(input);
            let loss = (output - target).norm(); // L2 loss
            total_loss += loss;
        }
        total_loss / inputs.len() as f64
    }

    fn update_parameters(&mut self, learning_rate: f64, gradients: &DMatrix<f64>, regularization: &Regularization) {
        let gradient_update = learning_rate * gradients;
    
        match regularization {
            Regularization::L2(lambda) => self.h = &self.h - &gradient_update - *lambda * &self.h, // Dereference lambda
            Regularization::L1(lambda) => {
                let signum = self.h.map(|x| x.signum());
                self.h = &self.h - &gradient_update - *lambda * &signum; // Dereference lambda
            }
        }
    }
    

    fn triangular_integral(vector: &DVector<f64>) -> f64 {
        vector.iter().enumerate().fold(0.0, |acc, (i, &x)| acc + x * ((i + 1) as f64))
    }
}

fn main() {
    let mut hextral = Hextral::new(0.1, 0.2);

    // Generate more populated vectors for inputs and targets
    let num_samples = 1000;
    let inputs: Vec<DVector<f64>> = (0..num_samples)
        .map(|_| DVector::from_iterator(10, (0..10).map(|_| rand::random::<f64>())))
        .collect();

    let targets: Vec<DVector<f64>> = (0..num_samples)
        .map(|_| DVector::from_iterator(10, (0..10).map(|_| rand::random::<f64>())))
        .collect();

    hextral.train(&inputs, &targets, 0.01, Regularization::L2(0.001), 100);

    // Test with a single input vector
    let input = DVector::from_iterator(10, (0..10).map(|_| rand::random::<f64>()));
    let prediction = hextral.predict(&input);
    println!("Prediction: {:?}", prediction);

    // Evaluate the trained model
    let evaluation_loss = hextral.evaluate(&inputs, &targets);
    println!("Evaluation Loss: {}", evaluation_loss);
}
