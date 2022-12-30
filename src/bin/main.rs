use deep_neural_network::{
  dnn::DeepNeuralNetwork,
  utils::*
};

fn main(){
    let layer_dims: Vec<u8>= [2,3,4].to_vec();
    let learning_rate: f32 = 0.2;
    let network = DeepNeuralNetwork{layer_dims, learning_rate};
    sigmoid();
}