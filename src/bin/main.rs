use std::{collections::HashMap, fs::OpenOptions};

use deep_neural_network::{dnn::DeepNeuralNetwork, utils::*};
use ndarray::{ Array2};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Parameter {
    parameter: HashMap<String, Array2<f32>>,
}


fn main() {
    let layer_dims: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate: f32 = 0.0075;
    let network = DeepNeuralNetwork {
        layer_dims,
        learning_rate,
    };
   
    let (x_train_data, y_train_data) = load_data_as_dataframe("datasets/training_set.csv");
    let (x_test_data, y_test_data) = load_data_as_dataframe("datasets/test_set.csv");

    let x_train_data_array = array_from_dataframe(&x_train_data) / 255.0;
    let y_train_data_array = array_from_dataframe(&y_train_data);

    let x_test_data_array = array_from_dataframe(&x_test_data) / 255.0;
    let y_test_data_array = array_from_dataframe(&y_test_data);

    let parameters = network.initialize_parameters();

    let iterations: usize = 500;

    let parameters = network.train_model(x_train_data_array, y_train_data_array, parameters, iterations, learning_rate);

    write_parameters_to_json_file(&parameters, "weights.json");

    // let parameters = load_weights_from_json();

    // println!("{:?}", js);

    let score = network.predict(&x_test_data_array, &y_test_data_array, &parameters);

    println!("{}", score);
}