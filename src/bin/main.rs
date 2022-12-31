use deep_neural_network::{
    dnn::{DeepNeuralNetwork, Matrix},
    utils::*,
};
use ndarray::s;

// use hdf5::{File, Result};
fn main() {
    let layer_dims: Vec<usize> = [2, 3, 4, 5].to_vec();
    let learning_rate: f32 = 0.2;
    let network = DeepNeuralNetwork {
        layer_dims: layer_dims.clone(),
        learning_rate,
    };
    let mut parameters = network.initialize_parameters();

    // match &parameters["W1"]{
    //   Matrix::Weight(matrix)=> {}
    //   Matrix::Bias(vector)=> {},
    // }

    for l in 1..layer_dims.len() {

        let weight_string = ["W", &l.to_string()].join("").to_string();
        let bias_string = ["b", &l.to_string()].join("").to_string();

        match &parameters[&weight_string] {
            Matrix::Weight(matrix) => println!("{} = {}",weight_string, matrix),
           _ => {},
        }

        match &parameters[&bias_string] {
          Matrix::Bias(vector) => {println!("{} = {}",bias_string, vector);}
          _ => {},
      }
    }

    // let file = File::open("datasets/test_catvnoncat.h5");
    // let test_data = file.dataset("datasets/test_catvnoncat.h5");
}
