use hextral::*;
use nalgebra::DVector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quaternion Activation Function Demo");
    println!("===================================");

    // Test quaternion normalization with a 4D vector (w, x, y, z)
    let input = DVector::from_vec(vec![3.0, 4.0, 0.0, 0.0]); // Should normalize to unit quaternion
    let activation = ActivationFunction::Quaternion;
    let output = activation.apply(&input);
    
    println!("Input quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", input[0], input[1], input[2], input[3]);
    println!("Normalized: [{:.3}, {:.3}, {:.3}, {:.3}]", output[0], output[1], output[2], output[3]);
    
    let norm = (output[0]*output[0] + output[1]*output[1] + output[2]*output[2] + output[3]*output[3]).sqrt();
    println!("Output norm: {:.6} (should be ~1.0)", norm);
    
    // Test with 8D vector (two quaternions)
    let input8 = DVector::from_vec(vec![1.0, 2.0, 2.0, 0.0, 5.0, 0.0, 0.0, 12.0]);
    let output8 = activation.apply(&input8);
    
    println!("\n8D input (2 quaternions):");
    println!("Q1: [{:.1}, {:.1}, {:.1}, {:.1}] -> [{:.3}, {:.3}, {:.3}, {:.3}]", 
             input8[0], input8[1], input8[2], input8[3],
             output8[0], output8[1], output8[2], output8[3]);
    println!("Q2: [{:.1}, {:.1}, {:.1}, {:.1}] -> [{:.3}, {:.3}, {:.3}, {:.3}]", 
             input8[4], input8[5], input8[6], input8[7],
             output8[4], output8[5], output8[6], output8[7]);
    
    let norm1 = (output8[0]*output8[0] + output8[1]*output8[1] + output8[2]*output8[2] + output8[3]*output8[3]).sqrt();
    let norm2 = (output8[4]*output8[4] + output8[5]*output8[5] + output8[6]*output8[6] + output8[7]*output8[7]).sqrt();
    println!("Norms: Q1={:.6}, Q2={:.6}", norm1, norm2);
    
    // Test in neural network
    println!("\nTesting in neural network:");
    let nn = Hextral::new(4, &[8], 4, ActivationFunction::Quaternion, Optimizer::adam(0.01));
    let prediction = nn.predict(&input).await;
    println!("Network output: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             prediction[0], prediction[1], prediction[2], prediction[3]);
    
    Ok(())
}